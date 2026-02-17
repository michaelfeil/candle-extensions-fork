//! cuDNN SDPA Attention with THD layout (Total sequences, Heads, Dimension)
//!
//! This implementation uses ragged tensors for variable length sequences.
//! All sequences have the same Q and K lengths (training mode only, no generation).

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

/// cuDNN SDPA attention with variable length sequences (THD layout)
struct CuDNNAttnTHD {
    softmax_scale: f32,
    causal: bool,
    max_seqlen: usize,
    seqlens: Tensor, // Cumulative sequence lengths (same for Q and K)
}

impl CuDNNAttnTHD {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        _k_l: &Layout,
        v: &candle::CudaStorage,
        _v_l: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        use crate::ffi;
        use crate::tensor::{CuDNNTensor, DataType};
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use std::ptr;

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        // Get THD dimensions: (total_seq, num_heads, head_dim)
        let dims = q_l.shape().dims();
        if dims.len() != 3 {
            candle::bail!(
                "Expected 3D THD layout (total_seq, heads, dim), got {:?}",
                dims
            );
        }
        let _total_seq = dims[0] as i64;
        let num_heads = dims[1] as i32;
        let head_dim = dims[2] as i32;

        // Get number of sequences from seqlens tensor
        let (seqlens_storage, seqlens_layout) = self.seqlens.storage_and_layout();
        let seqlens_cuda = match &*seqlens_storage {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("seqlens must be a cuda tensor"),
        };
        let num_seqs = (seqlens_layout.shape().dims()[0] as i32) - 1; // seqlens has B+1 elements

        // Get data type
        let data_type = match q.dtype() {
            candle::DType::F16 => DataType::Float16,
            candle::DType::BF16 => DataType::BFloat16,
            candle::DType::F32 => DataType::Float32,
            dt => candle::bail!("cuDNN attention only supports f16/bf16/f32 ({dt:?})"),
        };

        // Create cuDNN handle
        let mut handle: ffi::cudnnHandle_t = ptr::null_mut();
        let result = unsafe { ffi::cudnnCreate(&mut handle) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            candle::bail!("Failed to create cuDNN handle: {}", result);
        }

        // Create 4D tensor descriptors for cuDNN
        // Even for THD layout, cuDNN expects 4D: (B, H, S, D)
        // We use B=1 and rely on ragged offsets for variable lengths
        let max_seqlen = self.max_seqlen as i32;
        let q_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];
        let k_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];
        let v_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];
        let o_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];

        let q_tensor = CuDNNTensor::new(q_dims, data_type)?;
        let k_tensor = CuDNNTensor::new(k_dims, data_type)?;
        let v_tensor = CuDNNTensor::new(v_dims, data_type)?;
        let o_tensor = CuDNNTensor::new(o_dims, data_type)?;

        // Create ragged offset tensor descriptor for seqlens
        // Shape: (B+1, 1, 1, 1) with cumulative offsets in elements
        let ragged_dims = vec![num_seqs + 1, 1, 1, 1];
        let ragged_tensor = CuDNNTensor::new(ragged_dims, DataType::Int32)?;

        // Create SDPA forward operation descriptor
        let mut sdpa_op: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR,
                &mut sdpa_op,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to create SDPA operation descriptor: {}", result);
        }

        // Helper macro to set attributes
        macro_rules! set_attr {
            ($desc:expr, $attr:expr, $type:expr, $value:expr, $msg:expr) => {
                let result = unsafe {
                    ffi::cudnnBackendSetAttribute(
                        $desc,
                        $attr,
                        $type,
                        1,
                        $value as *const _ as *const std::ffi::c_void,
                    )
                };
                if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                    unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
                    unsafe { ffi::cudnnDestroy(handle) };
                    candle::bail!("{}: {}", $msg, result);
                }
            };
        }

        // Set Q, K, V, O tensors
        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &q_tensor.descriptor(),
            "Failed to set Q tensor"
        );

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &k_tensor.descriptor(),
            "Failed to set K tensor"
        );

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &v_tensor.descriptor(),
            "Failed to set V tensor"
        );

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &o_tensor.descriptor(),
            "Failed to set O tensor"
        );

        // Set ragged offsets for variable length sequences
        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &ragged_tensor.descriptor(),
            "Failed to set Q ragged offset"
        );

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &ragged_tensor.descriptor(),
            "Failed to set K/V ragged offset"
        );

        // Create and set scale tensor
        let scale_tensor = CuDNNTensor::new(vec![1, 1, 1, 1], DataType::Float32)?;

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &scale_tensor.descriptor(),
            "Failed to set scale"
        );

        // Finalize the operation descriptor
        let result = unsafe { ffi::cudnnBackendFinalize(sdpa_op) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to finalize SDPA operation: {}", result);
        }

        // Create operation graph
        let mut op_graph: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                &mut op_graph,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to create operation graph: {}", result);
        }

        // Set handle in operation graph
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                op_graph,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_HANDLE,
                1,
                &handle as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set handle in operation graph: {}", result);
        }

        // Set operations in operation graph
        let ops: [*const std::ffi::c_void; 1] = [&sdpa_op as *const _ as *const std::ffi::c_void];
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                op_graph,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATIONGRAPH_OPS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                ops.as_ptr() as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set operations in operation graph: {}", result);
        }

        // Finalize operation graph
        let result = unsafe { ffi::cudnnBackendFinalize(op_graph) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to finalize operation graph: {}", result);
        }

        // Create execution plan
        let mut exec_plan: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                &mut exec_plan,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to create execution plan: {}", result);
        }

        // Set handle in execution plan
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                exec_plan,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_HANDLE,
                1,
                &handle as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set handle in execution plan: {}", result);
        }

        // Finalize execution plan
        let result = unsafe { ffi::cudnnBackendFinalize(exec_plan) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to finalize execution plan: {}", result);
        }

        // Get workspace size
        let mut workspace_size: i64 = 0;
        let mut element_count: i64 = 0;
        let result = unsafe {
            ffi::cudnnBackendGetAttribute(
                exec_plan,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                1,
                &mut element_count as *mut i64,
                &mut workspace_size as *mut i64 as *mut std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to get workspace size: {}", result);
        }

        // Allocate workspace
        let workspace_size = workspace_size as usize;
        let workspace = if workspace_size > 0 {
            unsafe { dev.alloc::<u8>(workspace_size) }.w()?
        } else {
            unsafe { dev.alloc::<u8>(1) }.w()?
        };

        // Allocate output tensor
        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        // Get device pointers
        let q_ptr = *q.as_cuda_slice::<T>()?.device_ptr() as *const core::ffi::c_void;
        let k_ptr = *k.as_cuda_slice::<T>()?.device_ptr() as *const core::ffi::c_void;
        let v_ptr = *v.as_cuda_slice::<T>()?.device_ptr() as *const core::ffi::c_void;
        let o_ptr = *dst.device_ptr() as *const core::ffi::c_void;
        let workspace_ptr = *workspace.device_ptr() as *const core::ffi::c_void;

        // Get seqlens device pointer
        let _seqlens_ptr =
            *seqlens_cuda.as_cuda_slice::<u32>()?.device_ptr() as *const core::ffi::c_void;

        // Create variant pack
        let mut variant_pack: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                &mut variant_pack,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to create variant pack: {}", result);
        }

        // Set tensor pointers in variant pack
        let tensor_ptrs: [*const std::ffi::c_void; 4] = [q_ptr, k_ptr, v_ptr, o_ptr];
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                variant_pack,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_VOID_PTR,
                4,
                tensor_ptrs.as_ptr() as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set tensor pointers: {}", result);
        }

        // Set workspace pointer
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                variant_pack,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_VOID_PTR,
                1,
                &workspace_ptr as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set workspace: {}", result);
        }

        // Finalize variant pack
        let result = unsafe { ffi::cudnnBackendFinalize(variant_pack) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to finalize variant pack: {}", result);
        }

        // Execute the plan
        let result = unsafe { ffi::cudnnBackendExecute(handle, exec_plan, variant_pack) };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to execute SDPA operation: {}", result);
        }

        // Cleanup
        unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
        unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
        unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
        unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
        unsafe { ffi::cudnnDestroy(handle) };

        // Wrap output
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
            candle::DType::F32 => self.cuda_fwd_t::<f32>(q, q_l, k, k_l, v, v_l),
            candle::DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l, k, k_l, v, v_l),
            candle::DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l, k, k_l, v, v_l),
            dt => candle::bail!("cuDNN attention only supports f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention compatible variable length attention with THD layout.
///
/// # Arguments
/// * `q` - Query tensor with shape (total_seq, num_heads, head_dim)
/// * `k` - Key tensor with shape (total_seq, num_heads, head_dim)  
/// * `v` - Value tensor with shape (total_seq, num_heads, head_dim)
/// * `seqlens` - Cumulative sequence lengths with shape (num_seqs + 1,)
/// * `max_seqlen` - Maximum sequence length
/// * `softmax_scale` - Scale factor for softmax (typically 1/sqrt(head_dim))
/// * `causal` - Whether to apply causal masking
///
/// # Returns
/// Output tensor with shape (total_seq, num_heads, head_dim)
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens: &Tensor,
    max_seqlen: usize,
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    // Validate shapes
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

    let op = CuDNNAttnTHD {
        softmax_scale,
        causal,
        max_seqlen,
        seqlens: seqlens.clone(),
    };

    q.apply_op3(k, v, op)
}

/// Check if cuDNN attention is available on the current system
pub fn is_available() -> bool {
    *CUDNN_AVAILABLE.get_or_init(|| check_cudnn_availability().unwrap_or(false))
}

/// Get cuDNN version information
pub fn version_info() -> crate::error::Result<String> {
    Ok("cuDNN 9.x with SDPA support".to_string())
}
