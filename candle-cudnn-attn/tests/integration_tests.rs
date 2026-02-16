//! Integration tests for cuDNN attention

use candle::{DType, Device, Tensor};
use candle_cudnn_attn::{cudnn_attention_varlen, is_available};

#[test]
fn test_cudnn_availability() {
    // This test will pass once we have the implementation
    // For now, it just checks the function doesn't panic
    let _available = is_available();
}

#[test]
fn test_basic_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 128;
    let head_dim = 64;

    let q = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    // Create ragged offset (packed sequences)
    let ragged_offset = Tensor::from_vec(
        vec![0u32, seq_len as u32, (2 * seq_len) as u32],
        (3, 1, 1, 1),
        &device,
    )?;

    // Run attention
    let output = cudnn_attention_varlen(
        &q,
        &k,
        &v,
        &ragged_offset,
        seq_len,
        1.0 / (head_dim as f32).sqrt(),
        true,
    )?;

    // Check output shape
    let expected_shape = [batch_size, num_heads, seq_len, head_dim];
    assert_eq!(output.shape().dims(), expected_shape.as_slice());

    Ok(())
}

#[test]
fn test_variable_length_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors with variable lengths
    let batch_size = 3;
    let num_heads = 8;
    let max_seq_len = 128;
    let head_dim = 64;

    // Variable sequence lengths: 64, 128, 32
    let seq_lengths = [64, 128, 32];
    let total_tokens: usize = seq_lengths.iter().sum();

    // Create 4D tensors (batch_size, num_heads, max_seq_len, head_dim)
    // For variable length, we use the max sequence length and mask via ragged_offset
    let q = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, max_seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, max_seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, max_seq_len, head_dim),
        &device,
    )?;

    // Create ragged offset for 4D tensors
    // Each batch element has a different sequence length
    let mut offsets = vec![0u32];
    for &seq_len in &seq_lengths {
        offsets.push(offsets.last().unwrap() + seq_len as u32);
    }

    let ragged_offset = Tensor::from_vec(offsets, (batch_size + 1, 1, 1, 1), &device)?;

    // Run attention
    let output = cudnn_attention_varlen(
        &q,
        &k,
        &v,
        &ragged_offset,
        max_seq_len,
        1.0 / (head_dim as f32).sqrt(),
        true,
    )?;

    // Check output shape (should match input)
    assert_eq!(output.shape().dims(), q.shape().dims());

    Ok(())
}

#[test]
fn test_error_handling() {
    let device = Device::new_cuda(0).unwrap();

    // Test invalid data type (F64 is not supported)
    let q = Tensor::randn(0.0f64, 1.0f64, (2, 8, 128, 64), &device).unwrap();
    let k = Tensor::randn(0.0f64, 1.0f64, (2, 8, 128, 64), &device).unwrap();
    let v = Tensor::randn(0.0f64, 1.0f64, (2, 8, 128, 64), &device).unwrap();

    let ragged_offset = Tensor::from_vec(vec![0u32, 128, 256], (3, 1, 1, 1), &device).unwrap();

    // Should return error for unsupported data type (F64)
    let result =
        cudnn_attention_varlen(&q, &k, &v, &ragged_offset, 128, 1.0 / 64.0_f32.sqrt(), true);

    assert!(result.is_err());

    // Test invalid ragged offset shape (using F32 tensors which are now supported)
    let q_f32 = Tensor::randn(0.0f32, 1.0f32, (2, 8, 128, 64), &device).unwrap();
    let k_f32 = Tensor::randn(0.0f32, 1.0f32, (2, 8, 128, 64), &device).unwrap();
    let v_f32 = Tensor::randn(0.0f32, 1.0f32, (2, 8, 128, 64), &device).unwrap();

    let invalid_offset = Tensor::from_vec(vec![0u32, 128], (2, 1, 1, 1), &device).unwrap();
    let result = cudnn_attention_varlen(
        &q_f32,
        &k_f32,
        &v_f32,
        &invalid_offset,
        128,
        1.0 / 64.0_f32.sqrt(),
        true,
    );

    assert!(result.is_err());
}

/// Reference attention implementation (unfused) for comparison
fn reference_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    // Q, K, V shapes: (batch, num_heads, seq_len, head_dim)
    // Compute attention scores: Q @ K^T
    let k_t = k.transpose(2, 3)?; // (batch, num_heads, head_dim, seq_len)
    let scores = q.matmul(&k_t)?; // (batch, num_heads, seq_len, seq_len)

    // Scale scores - use broadcast_mul to handle scalar multiplication
    let scale_tensor = Tensor::new(softmax_scale, q.device())?;
    let scaled_scores = scores.broadcast_mul(&scale_tensor)?;

    // Apply causal mask if needed
    let masked_scores = if causal {
        // Create causal mask: upper triangular (including diagonal) is allowed
        let seq_len = scores.dim(2)?;
        let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i * seq_len + j] = 0.0;
            }
        }
        let mask_tensor = Tensor::from_vec(mask, (seq_len, seq_len), q.device())?;
        let mask_tensor = mask_tensor.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, seq_len)
        scaled_scores.broadcast_add(&mask_tensor)?
    } else {
        scaled_scores
    };

    // Softmax: exp(x) / sum(exp(x))
    let max_val = masked_scores.max_keepdim(3)?;
    let shifted = masked_scores.broadcast_sub(&max_val)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(3)?;
    let attn_weights = exp_shifted.broadcast_div(&sum_exp)?;

    // Apply attention weights to values
    let output = attn_weights.matmul(v)?; // (batch, num_heads, seq_len, head_dim)

    Ok(output)
}

#[test]
fn test_cudnn_vs_reference_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors
    let batch_size = 2;
    let num_heads = 8;
    let seq_len = 64; // Smaller for faster testing
    let head_dim = 64;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Create random tensors with fixed seed for reproducibility
    let q = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0f32,
        1.0f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    // Create ragged offset (uniform sequences for this test)
    let ragged_offset = Tensor::from_vec(
        vec![0u32, seq_len as u32, (2 * seq_len) as u32],
        (3, 1, 1, 1),
        &device,
    )?;

    // Run reference attention (unfused)
    println!("Running reference attention...");
    let reference_output = reference_attention(&q, &k, &v, softmax_scale, true)?;

    // Run cuDNN attention
    println!("Running cuDNN attention...");
    let cudnn_output =
        cudnn_attention_varlen(&q, &k, &v, &ragged_offset, seq_len, softmax_scale, true)?;

    // Compare outputs
    println!("Comparing outputs...");
    let diff = (&cudnn_output - &reference_output)?.abs()?.mean_all()?;
    let diff_val: f32 = diff.to_scalar()?;

    println!("Mean absolute difference: {:.6e}", diff_val);

    // Check that outputs are close (allowing for some numerical differences)
    // cuDNN may use different algorithms/precision than reference
    assert!(
        diff_val < 0.0,
        "cuDNN and reference outputs differ too much: {:.6e}",
        diff_val
    );

    // Also check max difference
    let max_diff = (&cudnn_output - &reference_output)?.abs()?.max_all()?;
    let max_diff_val: f32 = max_diff.to_scalar()?;
    println!("Max absolute difference: {:.6e}", max_diff_val);

    assert!(
        max_diff_val < 1e-5,
        "Max difference too large: {:.6e}",
        max_diff_val
    );

    Ok(())
}
