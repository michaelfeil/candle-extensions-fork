//! cuDNN SDPA Attention with THD layout (Total sequences, Heads, Dimension)
//!
//! Native THD varlen cuDNN backend path.

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

        if !q_l.is_contiguous() || !_k_l.is_contiguous() || !_v_l.is_contiguous() {
            candle::bail!("Q/K/V tensors must be contiguous for cuDNN THD attention");
        }

        // Input is THD: (total_tokens, num_heads, head_dim)
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

        // Decode cumulative sequence lengths.
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

        let mut seq_lens = Vec::with_capacity(cu_seqlens.len() - 1);
        for w in cu_seqlens.windows(2) {
            if w[1] < w[0] {
                candle::bail!("seqlens must be non-decreasing, got {:?}", cu_seqlens);
            }
            let len = w[1] - w[0];
            if len > self.max_seqlen as u32 {
                candle::bail!(
                    "seqlen {} exceeds max_seqlen {}",
                    len,
                    self.max_seqlen
                );
            }
            seq_lens.push(len);
        }
        let num_seqs = (cu_seqlens.len() as i64) - 1;
        if num_seqs <= 0 {
            candle::bail!("seqlens must contain at least 2 elements");
        }
        if cu_seqlens[cu_seqlens.len() - 1] as i64 != total_tokens {
            candle::bail!(
                "seqlens must end at total tokens {} (got {})",
                total_tokens,
                cu_seqlens[cu_seqlens.len() - 1]
            );
        }

        // Get data type
        let data_type = match q.dtype() {
            candle::DType::F16 => DataType::Float16,
            candle::DType::BF16 => DataType::BFloat16,
            dt => candle::bail!("cuDNN attention only supports f16/bf16 ({dt:?})"),
        };
        let _causal = self.causal;

        // Create cuDNN handle
        let mut handle: ffi::cudnnHandle_t = ptr::null_mut();
        let result = unsafe { ffi::cudnnCreate(&mut handle) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            candle::bail!("Failed to create cuDNN handle: {}", result);
        }

        // Build BHSD descriptors mapped by ragged offsets from THD packed inputs.
        let max_seqlen = self.max_seqlen as i64;
        let q_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];
        let kv_dims = vec![num_seqs, num_heads, max_seqlen, head_dim];
        // Packed THD storage with logical BHSD dims uses BSHD strides.
        let bhsd_strides = vec![
            max_seqlen * num_heads * head_dim, // B
            head_dim,                          // H
            num_heads * head_dim,              // S
            1,                                 // D
        ];

        const UID_Q: i64 = 101;
        const UID_K: i64 = 102;
        const UID_V: i64 = 103;
        const UID_O: i64 = 104;
        const UID_SEQ_LEN_Q: i64 = 105;
        const UID_SEQ_LEN_KV: i64 = 106;
        const UID_SCALE: i64 = 107;
        const UID_Q_RAGGED: i64 = 109;
        const UID_K_RAGGED: i64 = 110;
        const UID_V_RAGGED: i64 = 111;
        const UID_O_RAGGED: i64 = 112;

        // Ragged offsets in element units for THD packed layout.
        let q_ragged_offsets: Vec<i64> = cu_seqlens
            .iter()
            .map(|x| (*x as i64) * num_heads * head_dim)
            .collect();
        let k_ragged_offsets: Vec<i64> = cu_seqlens
            .iter()
            .map(|x| (*x as i64) * num_heads * head_dim)
            .collect();
        let v_ragged_offsets: Vec<i64> = cu_seqlens
            .iter()
            .map(|x| (*x as i64) * num_heads * head_dim)
            .collect();
        let o_ragged_offsets: Vec<i64> = cu_seqlens
            .iter()
            .map(|x| (*x as i64) * num_heads * head_dim)
            .collect();

        let q_ragged_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs + 1, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int64,
            UID_Q_RAGGED,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create q ragged descriptor: {e}")))?;
        let k_ragged_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs + 1, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int64,
            UID_K_RAGGED,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create k ragged descriptor: {e}")))?;
        let v_ragged_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs + 1, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int64,
            UID_V_RAGGED,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create v ragged descriptor: {e}")))?;
        let o_ragged_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs + 1, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int64,
            UID_O_RAGGED,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create o ragged descriptor: {e}")))?;
        let seqlen_q_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int32,
            UID_SEQ_LEN_Q,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create seq_len_q descriptor: {e}")))?;
        let seqlen_kv_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![num_seqs, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Int32,
            UID_SEQ_LEN_KV,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create seq_len_kv descriptor: {e}")))?;

        // Q/K/V/O use ragged offsets for THD varlen.
        let q_tensor = CuDNNTensor::new_with_uid_and_strides_and_ragged(
            q_dims.clone(),
            bhsd_strides.clone(),
            data_type,
            UID_Q,
            Some(q_ragged_tensor.descriptor()),
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create Q descriptor: {e}")))?;
        let k_tensor = CuDNNTensor::new_with_uid_and_strides_and_ragged(
            kv_dims.clone(),
            bhsd_strides.clone(),
            data_type,
            UID_K,
            Some(k_ragged_tensor.descriptor()),
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create K descriptor: {e}")))?;
        let v_tensor = CuDNNTensor::new_with_uid_and_strides_and_ragged(
            kv_dims,
            bhsd_strides.clone(),
            data_type,
            UID_V,
            Some(v_ragged_tensor.descriptor()),
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create V descriptor: {e}")))?;
        let o_tensor = CuDNNTensor::new_with_uid_and_strides_and_ragged(
            q_dims,
            bhsd_strides,
            data_type,
            UID_O,
            Some(o_ragged_tensor.descriptor()),
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create O descriptor: {e}")))?;

        // Create SDPA forward operation descriptor.
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

        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &seqlen_q_tensor.descriptor(),
            "Failed to set Q seqlen tensor"
        );
        set_attr!(
            sdpa_op,
            ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
            ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
            &seqlen_kv_tensor.descriptor(),
            "Failed to set KV seqlen tensor"
        );

        // Create and set scale tensor (runtime data pointer bound in variant pack).
        let scale_tensor = CuDNNTensor::new_with_uid_and_strides(
            vec![1, 1, 1, 1],
            vec![1, 1, 1, 1],
            DataType::Float32,
            UID_SCALE,
        )
        .map_err(|e| candle::Error::msg(format!("Failed to create scale descriptor: {e}")))?;

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
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                op_graph,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATIONGRAPH_OPS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &sdpa_op as *const _ as *const std::ffi::c_void,
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

        // Query engine configs from heuristics.
        let heur_modes = [
            ffi::cudnnBackendHeurMode_t_CUDNN_HEUR_MODE_INSTANT,
            ffi::cudnnBackendHeurMode_t_CUDNN_HEUR_MODE_A,
            ffi::cudnnBackendHeurMode_t_CUDNN_HEUR_MODE_B,
            ffi::cudnnBackendHeurMode_t_CUDNN_HEUR_MODE_FALLBACK,
        ];
        let mut selected_heur_desc: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let mut engine_config: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let mut last_status = ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS;
        let mut last_count: i64 = 0;

        for heur_mode in heur_modes {
            let mut heur_desc: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
            let result = unsafe {
                ffi::cudnnBackendCreateDescriptor(
                    ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
                    &mut heur_desc,
                )
            };
            if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                last_status = result;
                continue;
            }

            let result = unsafe {
                ffi::cudnnBackendSetAttribute(
                    heur_desc,
                    ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_ENGINEHEUR_MODE,
                    ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_HEUR_MODE,
                    1,
                    &heur_mode as *const _ as *const std::ffi::c_void,
                )
            };
            if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                last_status = result;
                unsafe { ffi::cudnnBackendDestroyDescriptor(heur_desc) };
                continue;
            }

            let result = unsafe {
                ffi::cudnnBackendSetAttribute(
                    heur_desc,
                    ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &op_graph as *const _ as *const std::ffi::c_void,
                )
            };
            if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                last_status = result;
                unsafe { ffi::cudnnBackendDestroyDescriptor(heur_desc) };
                continue;
            }

            let result = unsafe { ffi::cudnnBackendFinalize(heur_desc) };
            if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                last_status = result;
                unsafe { ffi::cudnnBackendDestroyDescriptor(heur_desc) };
                continue;
            }

            let mut engine_configs: Vec<ffi::cudnnBackendDescriptor_t> = vec![ptr::null_mut(); 8];
            let mut engine_config_count: i64 = 0;
            let result = unsafe {
                ffi::cudnnBackendGetAttribute(
                    heur_desc,
                    ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_ENGINEHEUR_RESULTS,
                    ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    engine_configs.len() as i64,
                    &mut engine_config_count as *mut i64,
                    engine_configs.as_mut_ptr() as *mut std::ffi::c_void,
                )
            };
            last_status = result;
            last_count = engine_config_count;
            if result == ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS
                && engine_config_count > 0
                && !engine_configs[0].is_null()
            {
                engine_config = engine_configs[0];
                selected_heur_desc = heur_desc;
                break;
            }

            unsafe { ffi::cudnnBackendDestroyDescriptor(heur_desc) };
        }

        if engine_config.is_null() {
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!(
                "Failed to query engine configs from heuristics: status={}, count={}",
                last_status,
                last_count
            );
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
            if !selected_heur_desc.is_null() {
                unsafe { ffi::cudnnBackendDestroyDescriptor(selected_heur_desc) };
            }
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
            if !selected_heur_desc.is_null() {
                unsafe { ffi::cudnnBackendDestroyDescriptor(selected_heur_desc) };
            }
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set handle in execution plan: {}", result);
        }

        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                exec_plan,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &engine_config as *const _ as *const std::ffi::c_void,
            )
        };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(selected_heur_desc) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set engine config in execution plan: {}", result);
        }

        // Finalize execution plan
        let result = unsafe { ffi::cudnnBackendFinalize(exec_plan) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(selected_heur_desc) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to finalize execution plan: {}", result);
        }
        if !selected_heur_desc.is_null() {
            unsafe { ffi::cudnnBackendDestroyDescriptor(selected_heur_desc) };
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

        let seq_lens_i32: Vec<i32> = seq_lens.into_iter().map(|x| x as i32).collect();
        let seqlen_q_dev = dev.htod_copy(seq_lens_i32.clone()).w()?;
        let seqlen_kv_dev = dev.htod_copy(seq_lens_i32).w()?;
        let q_ragged_dev = dev.htod_copy(q_ragged_offsets).w()?;
        let k_ragged_dev = dev.htod_copy(k_ragged_offsets).w()?;
        let v_ragged_dev = dev.htod_copy(v_ragged_offsets).w()?;
        let o_ragged_dev = dev.htod_copy(o_ragged_offsets).w()?;
        let seqlen_q_ptr = *seqlen_q_dev.device_ptr() as *const core::ffi::c_void;
        let seqlen_kv_ptr = *seqlen_kv_dev.device_ptr() as *const core::ffi::c_void;
        let q_ragged_ptr = *q_ragged_dev.device_ptr() as *const core::ffi::c_void;
        let k_ragged_ptr = *k_ragged_dev.device_ptr() as *const core::ffi::c_void;
        let v_ragged_ptr = *v_ragged_dev.device_ptr() as *const core::ffi::c_void;
        let o_ragged_ptr = *o_ragged_dev.device_ptr() as *const core::ffi::c_void;
        let scale = dev.htod_copy(vec![self.softmax_scale]).w()?;
        let scale_ptr = *scale.device_ptr() as *const core::ffi::c_void;

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

        // Set tensor pointers in variant pack.
        let tensor_ptrs: [*const std::ffi::c_void; 11] = [
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            seqlen_q_ptr,
            seqlen_kv_ptr,
            q_ragged_ptr,
            k_ragged_ptr,
            v_ragged_ptr,
            o_ragged_ptr,
            scale_ptr,
        ];
        let tensor_uids: [i64; 11] = [
            q_tensor.uid(),
            k_tensor.uid(),
            v_tensor.uid(),
            o_tensor.uid(),
            seqlen_q_tensor.uid(),
            seqlen_kv_tensor.uid(),
            q_ragged_tensor.uid(),
            k_ragged_tensor.uid(),
            v_ragged_tensor.uid(),
            o_ragged_tensor.uid(),
            scale_tensor.uid(),
        ];
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                variant_pack,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_VOID_PTR,
                tensor_ptrs.len() as i64,
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

        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                variant_pack,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                tensor_uids.len() as i64,
                tensor_uids.as_ptr() as *const std::ffi::c_void,
            )
        };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(variant_pack) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(exec_plan) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(op_graph) };
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            unsafe { ffi::cudnnDestroy(handle) };
            candle::bail!("Failed to set tensor unique IDs: {}", result);
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
            candle::DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l, k, k_l, v, v_l),
            candle::DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l, k, k_l, v, v_l),
            dt => candle::bail!("cuDNN attention only supports f16/bf16 ({dt:?})"),
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

/// Check if cuDNN attention is available on the current system
pub fn is_available() -> bool {
    *CUDNN_AVAILABLE.get_or_init(|| check_cudnn_availability().unwrap_or(false))
}

/// Get cuDNN version information
pub fn version_info() -> crate::error::Result<String> {
    Ok("cuDNN 9.x with SDPA support".to_string())
}
