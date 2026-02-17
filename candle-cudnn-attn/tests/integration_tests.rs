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

    // Create test tensors in THD format: (total_tokens, num_heads, head_dim)
    let num_heads = 8;
    let total_tokens = 256; // batch_size=2, seq_len=128 each
    let head_dim = 64;

    let q = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?;

    // Create cumulative sequence lengths for batch_size=2 with seq_len=128 each
    // seqlens = [0, 128, 256] means: sequence 0 has tokens [0, 128), sequence 1 has tokens [128, 256)
    let seqlens = Tensor::new(&[0u32, 128u32, 256u32], &device)?;

    // Run attention - expect error since cuDNN is not yet implemented
    let result = flash_attn_varlen(
        &q,
        &k,
        &v,
        &seqlens,
        128, // max_seqlen
        1.0 / (head_dim as f32).sqrt(),
        true, // causal
    );

    // Should fail loud since cuDNN is not available
    assert!(
        result.is_err(),
        "Expected error when cuDNN is not available"
    );

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

    // Create THD tensors: (total_tokens, num_heads, head_dim)
    let q = Tensor::randn(0.0f32, 1.0f32, (total_q, num_heads, head_dim), &device)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (total_k, num_heads, head_dim), &device)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (total_k, num_heads, head_dim), &device)?;

    // Create cumulative sequence lengths
    // For seq_lengths [64, 128, 32], cu_seqlens = [0, 64, 192, 224]
    let mut cu_seqlens = vec![0u32];
    for &seq_len in &seq_lengths {
        cu_seqlens.push(cu_seqlens.last().unwrap() + seq_len as u32);
    }

    let seqlens = Tensor::new(&cu_seqlens[..], &device)?;

    let max_seqlen = *seq_lengths.iter().max().unwrap();
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Run attention - expect error since cuDNN is not yet implemented
    let result = flash_attn_varlen(&q, &k, &v, &seqlens, max_seqlen, softmax_scale, true);

    // Should fail loud since cuDNN is not available
    assert!(
        result.is_err(),
        "Expected error when cuDNN is not available"
    );

    Ok(())
}

#[test]
fn test_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    // Test invalid data type (F64 is not supported)
    let q = Tensor::randn(0.0f64, 1.0f64, (256, 8, 64), &device)?;
    let k = Tensor::randn(0.0f64, 1.0f64, (256, 8, 64), &device)?;
    let v = Tensor::randn(0.0f64, 1.0f64, (256, 8, 64), &device)?;

    let seqlens = Tensor::new(&[0u32, 128u32, 256u32], &device)?;

    // Should return error for unsupported data type (F64)
    let result = flash_attn_varlen(&q, &k, &v, &seqlens, 128, 1.0 / 64.0_f32.sqrt(), true);

    assert!(result.is_err());

    // Test invalid seqlens shape (using F32 tensors which are supported)
    let q_f32 = Tensor::randn(0.0f32, 1.0f32, (256, 8, 64), &device)?;
    let k_f32 = Tensor::randn(0.0f32, 1.0f32, (256, 8, 64), &device)?;
    let v_f32 = Tensor::randn(0.0f32, 1.0f32, (256, 8, 64), &device)?;

    // Invalid seqlens (only 1 element, need at least 2)
    let invalid_seqlens = Tensor::new(&[0u32], &device)?;
    let result = flash_attn_varlen(
        &q_f32,
        &k_f32,
        &v_f32,
        &invalid_seqlens,
        128,
        1.0 / 64.0_f32.sqrt(),
        true,
    );

    assert!(result.is_err());

    Ok(())
}

/// Scalar FP32 reference for THD varlen attention.
///
/// This intentionally loops over every sequence slice, head, query index and key index,
/// mirroring the expected varlen behavior directly.
fn reference_attention_thd_index_loop(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cu_seqlens: &[u32],
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor> {
    let dims = q.dims3()?;
    let total_tokens = dims.0;
    let num_heads = dims.1;
    let head_dim = dims.2;

    let qf = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let kf = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let vf = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; total_tokens * num_heads * head_dim];

    let stride_t = num_heads * head_dim;
    let stride_h = head_dim;

    for b in 0..(cu_seqlens.len() - 1) {
        let start = cu_seqlens[b] as usize;
        let end = cu_seqlens[b + 1] as usize;
        let seq_len = end - start;

        for h in 0..num_heads {
            for i in 0..seq_len {
                let qi = start + i;

                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for (j, score) in scores.iter_mut().enumerate() {
                    if causal && j > i {
                        continue;
                    }
                    let kj = start + j;
                    let mut dot = 0f32;
                    let q_base = qi * stride_t + h * stride_h;
                    let k_base = kj * stride_t + h * stride_h;
                    for d in 0..head_dim {
                        dot += qf[q_base + d] * kf[k_base + d];
                    }
                    *score = dot * softmax_scale;
                }

                let mut max_score = f32::NEG_INFINITY;
                for &s in &scores {
                    if s > max_score {
                        max_score = s;
                    }
                }

                let mut sum_exp = 0f32;
                let mut probs = vec![0f32; seq_len];
                for j in 0..seq_len {
                    let p = (scores[j] - max_score).exp();
                    probs[j] = p;
                    sum_exp += p;
                }

                let out_base = qi * stride_t + h * stride_h;
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for j in 0..seq_len {
                        let kj = start + j;
                        let v_base = kj * stride_t + h * stride_h;
                        acc += (probs[j] / sum_exp) * vf[v_base + d];
                    }
                    out[out_base + d] = acc;
                }
            }
        }
    }

    Tensor::from_vec(out, (total_tokens, num_heads, head_dim), q.device())
}

/// Test flash_attn_varlen against reference implementation for THD layout
#[test]
fn test_flash_attn_varlen_thd() -> Result<(), Box<dyn std::error::Error>> {
    if !is_available() {
        eprintln!("Skipping test_flash_attn_varlen_thd because cuDNN is unavailable");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;

    // Test with F16
    println!("\nTesting THD layout with F16 (causal=false)...");
    test_flash_attn_varlen_thd_with_dtype(DType::F16, 2e-4, false, &device)?;

    println!("\nTesting THD layout with F16 (causal=true)...");
    test_flash_attn_varlen_thd_with_dtype(DType::F16, 2e-4, true, &device)?;

    Ok(())
}

/// Helper function to test flash_attn_varlen with THD layout
fn test_flash_attn_varlen_thd_with_dtype(
    dtype: DType,
    tolerance: f32,
    causal: bool,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create test tensors in THD format: (total_tokens, num_heads, head_dim)
    let num_heads = 4;
    let head_dim = 64;

    // Variable sequence lengths
    let seq_lengths = [31, 17, 23, 9];
    let batch_size = seq_lengths.len();
    let total_tokens: usize = seq_lengths.iter().sum();

    println!("  Batch size: {}", batch_size);
    println!("  Sequence lengths: {:?}", seq_lengths);
    println!("  Total tokens: {}", total_tokens);
    println!("  Num heads: {}", num_heads);
    println!("  Head dim: {}", head_dim);

    // Create THD tensors: (total_tokens, num_heads, head_dim)
    let q = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), device)?
        .to_dtype(dtype)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), device)?
        .to_dtype(dtype)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), device)?
        .to_dtype(dtype)?;

    // Create cumulative sequence lengths
    // For seq_lengths [31, 17, 23, 9], cu_seqlens = [0, 31, 48, 71, 80]
    let mut cu_seqlens = vec![0u32];
    for &seq_len in &seq_lengths {
        cu_seqlens.push(cu_seqlens.last().unwrap() + seq_len as u32);
    }

    let seqlens = Tensor::new(&cu_seqlens[..], device)?;

    let max_seqlen = *seq_lengths.iter().max().unwrap();
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    // Compute reference output using THD reference implementation
    println!("Computing reference output...");
    let reference_output =
        reference_attention_thd_index_loop(&q, &k, &v, &cu_seqlens, softmax_scale, causal)?;
    println!("  Reference output computed successfully");

    // Run cuDNN flash_attn_varlen
    println!("Running flash_attn_varlen...");
    let cudnn_output = flash_attn_varlen(&q, &k, &v, &seqlens, max_seqlen, softmax_scale, causal)?;

    println!("  cuDNN output computed successfully, comparing with reference...");
    let diff = (&cudnn_output.to_dtype(DType::F32)? - &reference_output)?
        .abs()?
        .mean_all()?;
    let diff_val: f32 = diff.to_scalar()?;

    println!("Mean absolute difference: {:.6e}", diff_val);
    println!("Tolerance: {:.6e}", tolerance);

    assert!(
        diff_val < tolerance,
        "cuDNN and reference outputs differ too much for {:?} causal={}: {:.6e} (tolerance: {:.6e})",
        dtype,
        causal,
        diff_val,
        tolerance
    );

    println!("  ✅ {:?} causal={} test passed!", dtype, causal);
    Ok(())
}
