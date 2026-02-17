//! High-level attention API using cuDNN frontend
//!
//! This module provides the main public API for cuDNN attention operations.
//! It follows the flash-attention-v1/v3 API for variable length sequences.

use crate::error::{CuDNNError, Result};
use candle::{DType, Shape, Tensor};
use std::sync::OnceLock;

static CUDNN_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if cuDNN is available and meets requirements
fn check_cudnn_availability() -> Result<bool> {
    // Try to create a cuDNN handle to verify it's working
    match crate::graph::CuDNNGraph::new() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Flash-attention compatible variable length attention.
///
/// This function provides the same API as flash-attention-v1/v3 for easy migration.
/// It implements scaled dot-product attention with variable length sequences.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(num_heads, total_q, head_size)`
/// * `k` - Key tensor with shape `(num_heads, total_k, head_size)`
/// * `v` - Value tensor with shape `(num_heads, total_k, head_size)`
/// * `seqlens_q` - Cumulative sequence lengths for queries with shape `(batch_size + 1,)`
/// * `seqlens_k` - Cumulative sequence lengths for keys/values with shape `(batch_size + 1,)`
/// * `max_seqlen_q` - Maximum query sequence length in the batch
/// * `max_seqlen_k` - Maximum key/value sequence length in the batch
/// * `softmax_scale` - Scale factor for attention scores (typically 1/sqrt(head_size))
/// * `causal` - Whether to apply causal masking
///
/// # Returns
///
/// Output tensor with shape `(num_heads, total_q, head_size)`
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    // Validate inputs
    validate_flash_attn_inputs(q, k, v, seqlens_q, seqlens_k)?;

    // Convert 3D tensors (num_heads, total, head_dim) to 4D (1, num_heads, total, head_dim)
    // by adding a batch dimension at the front
    let q_4d = q.unsqueeze(0)?; // (1, num_heads, total_q, head_dim)
    let k_4d = k.unsqueeze(0)?; // (1, num_heads, total_k, head_dim)
    let v_4d = v.unsqueeze(0)?; // (1, num_heads, total_k, head_dim)

    // Convert cu_seqlens (1D) to ragged_offset (4D)
    // cu_seqlens has shape (batch_size + 1,) with cumulative lengths
    // ragged_offset needs shape (batch_size + 1, 1, 1, 1)
    let batch_size = seqlens_q.dim(0)? - 1;
    let ragged_offset = seqlens_q
        .reshape((batch_size + 1, 1, 1, 1))?
        .to_dtype(DType::U32)?;

    // Validate ragged_offset with the correct batch size before conversion
    validate_ragged_offset(&ragged_offset, batch_size)?;

    // Use max of max_seqlen_q and max_seqlen_k for the implementation
    let max_seqlen = max_seqlen_q.max(max_seqlen_k);

    // Call the internal attention implementation
    let output_4d = attention_varlen_impl(
        &q_4d,
        &k_4d,
        &v_4d,
        &ragged_offset,
        max_seqlen,
        softmax_scale,
        causal,
    )?;

    // Convert output from 4D back to 3D
    // Output is (1, num_heads, total_q, head_dim) -> squeeze to (num_heads, total_q, head_dim)
    let output = output_4d.squeeze(0)?;

    Ok(output)
}

/// Internal implementation for variable length attention with 4D tensors
fn attention_varlen_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ragged_offset: &Tensor,
    _max_seqlen: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    // Validate inputs (ragged_offset was already validated in flash_attn_varlen)
    validate_attention_inputs_4d(q, k, v)?;

    // Check if cuDNN is actually available and working
    let available = check_cudnn_availability()?;

    if !available {
        return Err(CuDNNError::not_available(
            "cuDNN is not available on this system",
        ));
    }

    // Use cuDNN implementation - no fallback to reference
    cudnn_attention_impl(q, k, v, ragged_offset, softmax_scale, causal)
}

/// Validate flash-attention compatible inputs (3D tensors)
fn validate_flash_attn_inputs(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
) -> Result<()> {
    // Check that q, k, v are 3D tensors
    if q.dims().len() != 3 || k.dims().len() != 3 || v.dims().len() != 3 {
        return Err(CuDNNError::invalid_shape(
            "Q, K, V must be 3D tensors (num_heads, total, head_dim)",
            format!("Q: {:?}, K: {:?}, V: {:?}", q.shape(), k.shape(), v.shape()),
        ));
    }

    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();

    // Check that num_heads match
    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        return Err(CuDNNError::invalid_configuration(
            "Query, key, and value must have the same number of heads",
        ));
    }

    // Check that head_dim matches
    if q_dims[2] != k_dims[2] || q_dims[2] != v_dims[2] {
        return Err(CuDNNError::invalid_configuration(
            "Query, key, and value must have the same head dimension",
        ));
    }

    // Check that key and value have the same total length
    if k_dims[1] != v_dims[1] {
        return Err(CuDNNError::invalid_configuration(
            "Key and value must have the same total sequence length",
        ));
    }

    // Check head dimension is multiple of 8
    if q_dims[2] % 8 != 0 {
        return Err(CuDNNError::invalid_configuration(
            "Head dimension must be a multiple of 8",
        ));
    }

    // Check seqlens tensors are 1D
    if seqlens_q.dims().len() != 1 || seqlens_k.dims().len() != 1 {
        return Err(CuDNNError::invalid_shape(
            "seqlens_q and seqlens_k must be 1D tensors",
            format!(
                "seqlens_q: {:?}, seqlens_k: {:?}",
                seqlens_q.shape(),
                seqlens_k.shape()
            ),
        ));
    }

    // Check seqlens have the same length
    if seqlens_q.dim(0)? != seqlens_k.dim(0)? {
        return Err(CuDNNError::invalid_configuration(
            "seqlens_q and seqlens_k must have the same length",
        ));
    }

    // Check seqlens have at least 2 elements (batch_size >= 1)
    if seqlens_q.dim(0)? < 2 {
        return Err(CuDNNError::invalid_configuration(
            "seqlens_q must have at least 2 elements (batch_size >= 1)",
        ));
    }

    // Check data types
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(CuDNNError::invalid_configuration(
            "All input tensors must have the same data type",
        ));
    }

    match q.dtype() {
        DType::F16 | DType::BF16 | DType::F32 => {}
        _ => {
            return Err(CuDNNError::unsupported_data_type(format!(
                "{:?}",
                q.dtype()
            )));
        }
    }

    // Check device
    if !q.device().is_cuda() || !k.device().is_cuda() || !v.device().is_cuda() {
        return Err(CuDNNError::invalid_configuration(
            "All input tensors must be on CUDA device",
        ));
    }

    Ok(())
}

/// Validate attention input shapes for 4D tensors (internal use)
fn validate_attention_inputs_4d(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<()> {
    // Check dimensions
    if q.dims().len() != 4 || k.dims().len() != 4 || v.dims().len() != 4 {
        return Err(CuDNNError::invalid_shape(
            "All tensors must have 4 dimensions (B, H, S, D)",
            format!("Q: {:?}, K: {:?}, V: {:?}", q.shape(), k.shape(), v.shape()),
        ));
    }

    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();

    // Check batch size
    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        return Err(CuDNNError::invalid_configuration(
            "All tensors must have the same batch size",
        ));
    }

    // Check head dimensions
    if q_dims[3] != k_dims[3] {
        return Err(CuDNNError::invalid_configuration(
            "Query and key must have the same head dimension",
        ));
    }

    // Check sequence lengths for key/value
    if k_dims[2] != v_dims[2] {
        return Err(CuDNNError::invalid_configuration(
            "Key and value must have the same sequence length",
        ));
    }

    // Check head dimension is multiple of 8
    if q_dims[3] % 8 != 0 {
        return Err(CuDNNError::invalid_configuration(
            "Head dimension must be a multiple of 8",
        ));
    }

    // Check data types
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(CuDNNError::invalid_configuration(
            "All input tensors must have the same data type",
        ));
    }

    match q.dtype() {
        DType::F16 | DType::BF16 | DType::F32 => {}
        _ => {
            return Err(CuDNNError::unsupported_data_type(format!(
                "{:?}",
                q.dtype()
            )));
        }
    }

    // Check device
    if !q.device().is_cuda() || !k.device().is_cuda() || !v.device().is_cuda() {
        return Err(CuDNNError::invalid_configuration(
            "All input tensors must be on CUDA device",
        ));
    }

    Ok(())
}

/// Validate ragged offset tensor (for internal 4D use)
fn validate_ragged_offset(ragged_offset: &Tensor, batch_size: usize) -> Result<()> {
    let offset_shape = ragged_offset.shape();

    if offset_shape.dims() != &[batch_size + 1, 1, 1, 1] {
        return Err(CuDNNError::invalid_shape(
            format!("({}, 1, 1, 1)", batch_size + 1),
            format!("{:?}", offset_shape),
        )
        .into());
    }

    if ragged_offset.dtype() != DType::U32 {
        return Err(
            CuDNNError::invalid_configuration("Ragged offset must be U32 data type").into(),
        );
    }

    Ok(())
}

/// Check if cuDNN attention is available on the current system
pub fn is_available() -> bool {
    *CUDNN_AVAILABLE.get_or_init(|| check_cudnn_availability().unwrap_or(false))
}

/// Get cuDNN version information
pub fn version_info() -> Result<String> {
    // This will be implemented when we have the FFI bindings
    Ok("cuDNN version not yet implemented".to_string())
}

/// Actual cuDNN attention implementation (4D tensors)
fn cudnn_attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ragged_offset: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    use crate::ffi;
    use crate::graph::CuDNNGraph;
    use crate::tensor::{CuDNNTensor, DataType};
    use std::ptr;

    // Create cuDNN handle and graph
    let graph = CuDNNGraph::new()?;

    // Get tensor dimensions
    let q_dims: Vec<i32> = q.shape().dims().iter().map(|&d| d as i32).collect();
    let k_dims: Vec<i32> = k.shape().dims().iter().map(|&d| d as i32).collect();
    let v_dims: Vec<i32> = v.shape().dims().iter().map(|&d| d as i32).collect();

    // Get data type
    let data_type = DataType::from_candle(q.dtype())
        .ok_or_else(|| CuDNNError::unsupported_data_type(format!("{:?}", q.dtype())))?;

    // Create tensor descriptors
    let q_tensor = CuDNNTensor::new(q_dims, data_type)?;
    let k_tensor = CuDNNTensor::new(k_dims, data_type)?;
    let v_tensor = CuDNNTensor::new(v_dims, data_type)?;

    // For now, return an error since cuDNN SDPA graph building is not yet implemented
    // The reference implementation is available in tests for validation
    drop(q_tensor);
    drop(k_tensor);
    drop(v_tensor);
    drop(graph);

    Err(CuDNNError::internal(
        "cuDNN SDPA implementation not yet available.",
    ))
}
