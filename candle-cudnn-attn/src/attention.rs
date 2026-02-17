//! cuDNN SDPA Attention with THD layout (total_tokens, heads, dim)

use candle::cuda_backend::WrapErr;
use candle::{backend::BackendStorage, Layout, Shape, Tensor};
use std::sync::OnceLock;

static CUDNN_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn check_cudnn_availability() -> crate::error::Result<bool> {
    match crate::graph::CuDNNGraph::new() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

struct CuDNNAttnTHD {
    softmax_scale: f32,
    causal: bool,
    max_seqlen: usize,
    seqlens: Tensor,
}

impl CuDNNAttnTHD {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;

        if !q_l.is_contiguous() || !k_l.is_contiguous() || !v_l.is_contiguous() {
            candle::bail!("Q/K/V tensors must be contiguous for cuDNN THD attention");
        }

        let dims = q_l.shape().dims();
        if dims.len() != 3 {
            candle::bail!(
                "Expected 3D THD layout (total_tokens, heads, dim), got {:?}",
                dims
            );
        }
        let total_tokens = dims[0] as i64;
        let num_heads = dims[1] as i64;
        let head_dim = dims[2] as i64;

        if self.seqlens.shape().dims().len() != 1 {
            candle::bail!(
                "seqlens must be 1D with shape (batch + 1,), got {:?}",
                self.seqlens.shape().dims()
            );
        }
        let cu_seqlens = self.seqlens.to_vec1::<u32>()?;
        if cu_seqlens.len() < 2 {
            candle::bail!("seqlens must contain at least 2 elements");
        }
        if cu_seqlens[0] != 0 {
            candle::bail!("seqlens must start at 0, got {}", cu_seqlens[0]);
        }
        if cu_seqlens[cu_seqlens.len() - 1] as i64 != total_tokens {
            candle::bail!(
                "seqlens must end at total tokens {} (got {})",
                total_tokens,
                cu_seqlens[cu_seqlens.len() - 1]
            );
        }

        let mut seq_lens = Vec::with_capacity(cu_seqlens.len() - 1);
        for w in cu_seqlens.windows(2) {
            if w[1] < w[0] {
                candle::bail!("seqlens must be non-decreasing, got {:?}", cu_seqlens);
            }
            let len = w[1] - w[0];
            if len > self.max_seqlen as u32 {
                candle::bail!("seqlen {} exceeds max_seqlen {}", len, self.max_seqlen);
            }
            seq_lens.push(len as i32);
        }

        let is_bf16 = match q.dtype() {
            candle::DType::F16 => false,
            candle::DType::BF16 => true,
            dt => candle::bail!("cuDNN attention only supports f16/bf16 ({dt:?})"),
        };

        let num_seqs = (cu_seqlens.len() as i64) - 1;
        let q_ragged_offsets: Vec<i64> = cu_seqlens
            .iter()
            .map(|x| (*x as i64) * num_heads * head_dim)
            .collect();
        let k_ragged_offsets = q_ragged_offsets.clone();
        let v_ragged_offsets = q_ragged_offsets.clone();
        let o_ragged_offsets = q_ragged_offsets.clone();

        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let seq_q_dev = dev.htod_copy(seq_lens.clone()).w()?;
        let seq_kv_dev = dev.htod_copy(seq_lens).w()?;
        let q_ragged_dev = dev.htod_copy(q_ragged_offsets).w()?;
        let k_ragged_dev = dev.htod_copy(k_ragged_offsets).w()?;
        let v_ragged_dev = dev.htod_copy(v_ragged_offsets).w()?;
        let o_ragged_dev = dev.htod_copy(o_ragged_offsets).w()?;

        let q_ptr = *q.as_cuda_slice::<T>()?.device_ptr() as *mut core::ffi::c_void;
        let k_ptr = *k.as_cuda_slice::<T>()?.device_ptr() as *mut core::ffi::c_void;
        let v_ptr = *v.as_cuda_slice::<T>()?.device_ptr() as *mut core::ffi::c_void;
        let o_ptr = *dst.device_ptr() as *mut core::ffi::c_void;

        let seq_q_ptr = *seq_q_dev.device_ptr() as *const i32;
        let seq_kv_ptr = *seq_kv_dev.device_ptr() as *const i32;
        let q_ragged_ptr = *q_ragged_dev.device_ptr() as *const i64;
        let k_ragged_ptr = *k_ragged_dev.device_ptr() as *const i64;
        let v_ragged_ptr = *v_ragged_dev.device_ptr() as *const i64;
        let o_ragged_ptr = *o_ragged_dev.device_ptr() as *const i64;

        let stream_ptr = *dev.cu_stream() as *mut core::ffi::c_void;

        crate::frontend::run_thd_sdpa_fwd(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            seq_q_ptr,
            seq_kv_ptr,
            q_ragged_ptr,
            k_ragged_ptr,
            v_ragged_ptr,
            o_ragged_ptr,
            num_seqs,
            num_heads,
            self.max_seqlen as i64,
            head_dim,
            self.softmax_scale,
            self.causal,
            is_bf16,
            stream_ptr,
        )
        .map_err(candle::Error::msg)?;

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for CuDNNAttnTHD {
    fn name(&self) -> &'static str {
        "cudnn-attn-thd"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &Layout,
        _: &candle::CpuStorage,
        _: &Layout,
        _: &candle::CpuStorage,
        _: &Layout,
    ) -> candle::Result<(candle::CpuStorage, Shape)> {
        candle::bail!("cuDNN attention is only supported on CUDA")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l, k, k_l, v, v_l),
            candle::DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l, k, k_l, v, v_l),
            dt => candle::bail!("cuDNN attention only supports f16/bf16 ({dt:?})"),
        }
    }
}

pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens: &Tensor,
    max_seqlen: usize,
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();

    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
        candle::bail!(
            "Expected 3D THD layout (total_seq, heads, dim), got Q:{:?} K:{:?} V:{:?}",
            q_dims,
            k_dims,
            v_dims
        );
    }

    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        candle::bail!(
            "Total sequence length mismatch: Q={}, K={}, V={}",
            q_dims[0],
            k_dims[0],
            v_dims[0]
        );
    }

    if q_dims[1] != k_dims[1] || q_dims[1] != v_dims[1] {
        candle::bail!(
            "Number of heads mismatch: Q={}, K={}, V={}",
            q_dims[1],
            k_dims[1],
            v_dims[1]
        );
    }

    if q_dims[2] != k_dims[2] || q_dims[2] != v_dims[2] {
        candle::bail!(
            "Head dimension mismatch: Q={}, K={}, V={}",
            q_dims[2],
            k_dims[2],
            v_dims[2]
        );
    }

    if max_seqlen == 0 {
        candle::bail!("max_seqlen must be > 0");
    }

    let seqlens_dims = seqlens.shape().dims();
    if seqlens_dims.len() != 1 || seqlens_dims[0] < 2 {
        candle::bail!(
            "Expected seqlens shape (batch + 1,) with at least 2 elements, got {:?}",
            seqlens_dims
        );
    }
    if seqlens.dtype() != candle::DType::U32 {
        candle::bail!("seqlens must be U32, got {:?}", seqlens.dtype());
    }

    let op = CuDNNAttnTHD {
        softmax_scale,
        causal,
        max_seqlen,
        seqlens: seqlens.clone(),
    };
    q.apply_op3(k, v, op)
}

pub fn is_available() -> bool {
    *CUDNN_AVAILABLE.get_or_init(|| check_cudnn_availability().unwrap_or(false))
}

pub fn version_info() -> crate::error::Result<String> {
    Ok("cuDNN 9.x with SDPA support".to_string())
}
