//! Integration tests for cuDNN attention

use candle::{DType, Device, Tensor};
use candle_cudnn_attn::{flash_attn_varlen, is_available};

#[test]
fn test_cudnn_availability() {
    // This test will pass once we have the implementation
    // For now, it just checks the function doesn't panic
    let _available = is_available();
}

#[test]
fn test_basic_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors in 3D format: (num_heads, total_tokens, head_dim)
    let num_heads = 8;
    let total_tokens = 256; // batch_size=2, seq_len=128 each
    let head_dim = 64;

    let q = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;

    // Create cumulative sequence lengths for batch_size=2 with seq_len=128 each
    // seqlens = [0, 128, 256] means: sequence 0 has tokens [0, 128), sequence 1 has tokens [128, 256)
    let seqlens_q = Tensor::new(&[0u32, 128u32, 256u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 128u32, 256u32], &device)?;

    // Run attention
    let output = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens_q,
        &seqlens_k,
        128, // max_seqlen_q
        128, // max_seqlen_k
        1.0 / (head_dim as f32).sqrt(),
        true, // causal
    )?;

    // Check output shape
    let expected_shape = [num_heads, total_tokens, head_dim];
    assert_eq!(output.shape().dims(), expected_shape.as_slice());

    Ok(())
}

#[test]
fn test_variable_length_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors with variable lengths
    let num_heads = 8;
    let head_dim = 64;

    // Variable sequence lengths: 64, 128, 32
    let seq_lengths = [64, 128, 32];
    let total_q: usize = seq_lengths.iter().sum();
    let total_k: usize = total_q; // Same for simplicity

    // Create 3D tensors: (num_heads, total_tokens, head_dim)
    let q = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_q, head_dim), &device)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_k, head_dim), &device)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_k, head_dim), &device)?;

    // Create cumulative sequence lengths for variable lengths
    // For seq_lengths [64, 128, 32], cu_seqlens = [0, 64, 192, 224]
    let mut offsets = vec![0u32];
    for &seq_len in &seq_lengths {
        offsets.push(offsets.last().unwrap() + seq_len as u32);
    }

    let seqlens_q = Tensor::new(&offsets[..], &device)?;
    let seqlens_k = Tensor::new(&offsets[..], &device)?;

    let max_seqlen = *seq_lengths.iter().max().unwrap();

    // Run attention
    let output = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens_q,
        &seqlens_k,
        max_seqlen,
        max_seqlen,
        1.0 / (head_dim as f32).sqrt(),
        true,
    )?;

    // Check output shape
    assert_eq!(output.shape().dims(), &[num_heads, total_q, head_dim]);

    Ok(())
}

#[test]
fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Test invalid data type (F64 is not supported)
    let q = Tensor::randn(0.0f64, 1.0f64, (8, 128, 64), &device)?;
    let k = Tensor::randn(0.0f64, 1.0f64, (8, 128, 64), &device)?;
    let v = Tensor::randn(0.0f64, 1.0f64, (8, 128, 64), &device)?;

    let seqlens_q = Tensor::new(&[0u32, 128u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 128u32], &device)?;

    // Should return error for unsupported data type (F64)
    let result = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens_q,
        &seqlens_k,
        128,
        128,
        1.0 / 64.0_f32.sqrt(),
        true,
    );

    assert!(result.is_err());

    // Test invalid seqlens shape (using F32 tensors which are supported)
    let q_f32 = Tensor::randn(0.0f32, 1.0f32, (8, 128, 64), &device)?;
    let k_f32 = Tensor::randn(0.0f32, 1.0f32, (8, 128, 64), &device)?;
    let v_f32 = Tensor::randn(0.0f32, 1.0f32, (8, 128, 64), &device)?;

    // Invalid seqlens (only 1 element, need at least 2)
    let invalid_seqlens = Tensor::new(&[0u32], &device)?;
    let result = flash_attn_varlen(
        &q_f32,
        &k_f32,
        &v_f32,
        &invalid_seqlens,
        &invalid_seqlens,
        128,
        128,
        1.0 / 64.0_f32.sqrt(),
        true,
    );

    assert!(result.is_err());

    Ok(())
}

/// Reference attention implementation for 3D tensors (flash-attention compatible)
/// This is used for testing and as a fallback
fn reference_attention_3d(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    // Convert 3D to 4D by adding batch dimension
    let q_4d = q.unsqueeze(0)?;
    let k_4d = k.unsqueeze(0)?;
    let v_4d = v.unsqueeze(0)?;

    // Call 4D reference implementation
    let output_4d = reference_attention_4d(&q_4d, &k_4d, &v_4d, softmax_scale, causal)?;

    // Convert back to 3D
    output_4d.squeeze(0)
}

/// Reference attention implementation for 4D tensors (internal use)
fn reference_attention_4d(
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

    // Scale scores by softmax_scale
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
fn test_flash_attn_varlen_against_reference() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test tensors matching flash-attn-v1 test format
    // 3D tensors: (num_heads, total_tokens, head_dim)
    let num_heads = 2;
    let total_tokens = 6; // batch_size=1, seq_len=6 for simplicity
    let head_dim = 8;

    // Create simple test data using F32 to avoid dtype issues
    let q = Tensor::arange(0u32, (num_heads * total_tokens * head_dim) as u32, &device)?
        .to_dtype(DType::F32)?
        .reshape((num_heads, total_tokens, head_dim))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    // Cumulative sequence lengths: batch_size=1, sequence length=6
    let seqlens_q = Tensor::new(&[0u32, 6u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 6u32], &device)?;

    let softmax_scale = 0.5;

    // Run flash_attn_varlen
    let cudnn_output = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens_q,
        &seqlens_k,
        6,
        6,
        softmax_scale,
        false,
    )?;

    // Run reference implementation
    let reference_output = reference_attention_3d(&q, &k, &v, softmax_scale, false)?;

    // Compare outputs
    let diff = (&cudnn_output - &reference_output)?.abs()?.mean_all()?;
    let diff_val: f32 = diff.to_scalar()?;

    println!("Mean absolute difference: {:.6e}", diff_val);

    // Check that outputs are close
    assert!(
        diff_val < 1e-3,
        "cuDNN and reference outputs differ too much: {:.6e}",
        diff_val
    );

    Ok(())
}

/// Compute reference attention for variable length sequences
/// This processes each sequence independently and concatenates results
fn compute_varlen_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens: &[u32],
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    let num_heads = q.dim(0)?;
    let head_dim = q.dim(2)?;
    let batch_size = cu_seqlens.len() - 1;

    let mut outputs = Vec::new();

    // Process each sequence independently
    for i in 0..batch_size {
        let start_q = cu_seqlens[i] as usize;
        let end_q = cu_seqlens[i + 1] as usize;
        let seq_len_q = end_q - start_q;

        // For simplicity, assume k and v have the same sequence lengths as q
        let start_k = start_q;
        let end_k = end_q;
        let seq_len_k = seq_len_q;

        // Extract slices for this sequence
        // q_slice shape: (num_heads, seq_len_q, head_dim)
        let q_slice = q.narrow(1, start_q, seq_len_q)?;
        let k_slice = k.narrow(1, start_k, seq_len_k)?;
        let v_slice = v.narrow(1, start_k, seq_len_k)?;

        // Compute attention for this sequence using 4D reference
        // Add batch dimension: (1, num_heads, seq_len, head_dim)
        let q_4d = q_slice.unsqueeze(0)?;
        let k_4d = k_slice.unsqueeze(0)?;
        let v_4d = v_slice.unsqueeze(0)?;

        let out_4d = reference_attention_4d(&q_4d, &k_4d, &v_4d, softmax_scale, causal)?;
        let out_3d = out_4d.squeeze(0)?; // (num_heads, seq_len, head_dim)

        outputs.push(out_3d);
    }

    // Concatenate all sequence outputs along the sequence dimension (dim 1)
    Tensor::cat(&outputs, 1)
}

/// Test variable length attention with actual different sequence lengths
/// This test creates sequences of different lengths and verifies that
/// the attention is computed correctly for each sequence independently
#[test]
fn test_variable_length_non_uniform() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Create test with actual variable lengths
    let num_heads = 4;
    let head_dim = 64;

    // Different sequence lengths: 32, 64, 48
    let seq_lengths = [32, 64, 48];
    let batch_size = seq_lengths.len();
    let total_q: usize = seq_lengths.iter().sum();
    let total_k: usize = total_q;

    println!("Testing variable length attention:");
    println!("  Batch size: {}", batch_size);
    println!("  Sequence lengths: {:?}", seq_lengths);
    println!("  Total tokens: {}", total_q);

    // Create 3D tensors: (num_heads, total_tokens, head_dim)
    let q = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_q, head_dim), &device)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_k, head_dim), &device)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_k, head_dim), &device)?;

    // Create cumulative sequence lengths
    // For seq_lengths [32, 64, 48], cu_seqlens = [0, 32, 96, 144]
    let mut cu_seqlens = vec![0u32];
    for &seq_len in &seq_lengths {
        cu_seqlens.push(cu_seqlens.last().unwrap() + seq_len as u32);
    }

    let seqlens_q = Tensor::new(&cu_seqlens[..], &device)?;
    let seqlens_k = Tensor::new(&cu_seqlens[..], &device)?;

    let max_seqlen = *seq_lengths.iter().max().unwrap();
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Run flash_attn_varlen
    println!("Running flash_attn_varlen...");
    let cudnn_output = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens_q,
        &seqlens_k,
        max_seqlen,
        max_seqlen,
        softmax_scale,
        true,
    )?;

    // Compute reference output by processing each sequence independently
    println!("Computing reference output...");
    let reference_output = compute_varlen_reference(&q, &k, &v, &cu_seqlens, softmax_scale, true)?;

    // Compare outputs
    let diff = (&cudnn_output - &reference_output)?.abs()?.mean_all()?;
    let diff_val: f32 = diff.to_scalar()?;

    println!("Mean absolute difference: {:.6e}", diff_val);

    // Check that outputs are close
    assert!(
        diff_val < 1e-3,
        "cuDNN and reference outputs differ too much: {:.6e}",
        diff_val
    );

    println!("✅ Variable length test passed!");

    Ok(())
}
