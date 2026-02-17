//! Basic cuDNN attention example with flash-attention compatible API

use candle::{DType, Device, Tensor};
use candle_cudnn_attn::flash_attn_varlen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cuDNN Attention Example - Flash-Attention Compatible API");
    println!("========================================================");

    // Check availability
    if !candle_cudnn_attn::is_available() {
        println!("❌ cuDNN attention is not available on this system");
        println!("Requirements:");
        println!("  - CUDA 12.0 or newer");
        println!("  - cuDNN 9.0 or newer");
        println!("  - GPU with SM80 (Ampere) architecture or newer");
        return Ok(());
    }

    println!("✅ cuDNN attention is available");

    // Get CUDA device
    let device = Device::new_cuda(0)?;
    println!("📱 Using CUDA device: {:?}", device);

    // Create test tensors in 3D format (flash-attention compatible)
    let num_heads = 8;
    let batch_size = 4;
    let seq_len = 128;
    let head_dim = 64;
    let total_tokens = batch_size * seq_len;

    println!("\n📊 Tensor Configuration:");
    println!("  Number of heads: {}", num_heads);
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Head dimension: {}", head_dim);
    println!("  Total tokens: {}", total_tokens);

    // Create random tensors in THD format: (total_tokens, num_heads, head_dim)
    let q = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?
        .to_dtype(DType::F16)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?
        .to_dtype(DType::F16)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (total_tokens, num_heads, head_dim), &device)?
        .to_dtype(DType::F16)?;

    println!("✅ Created input tensors");

    // Create cumulative sequence lengths (cu_seqlens)
    // For batch_size=4 with uniform seq_len=128: [0, 128, 256, 384, 512]
    let mut seqlens = vec![0u32];
    for i in 1..=batch_size {
        seqlens.push((i * seq_len) as u32);
    }

    let seqlens_q = Tensor::new(&seqlens[..], &device)?;
    println!("✅ Created cumulative sequence lengths (cu_seqlens)");

    for causal in [false, true] {
        println!("\n🚀 Running cuDNN attention (causal={})...", causal);
        let mut output = flash_attn_varlen(
            &q,
            &k,
            &v,
            &seqlens_q,
            seq_len, // max_seqlen
            1.0 / (head_dim as f32).sqrt(),
            causal,
        )?;
        // Force completion of the warmup launch.
        let _ = output.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;

        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            output = flash_attn_varlen(
                &q,
                &k,
                &v,
                &seqlens_q,
                seq_len,
                1.0 / (head_dim as f32).sqrt(),
                causal,
            )?;
        }
        let _ = output.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        let duration = start.elapsed();
        println!(
            "✅ Steady-state avg over {} iters: {:?}",
            iters,
            duration / iters as u32
        );

        println!("\n📋 Output Verification:");
        println!("  Input shape: {:?}", q.shape());
        println!("  Output shape: {:?}", output.shape());

        let output_f32 = output.to_dtype(DType::F32)?;
        let output_mean = output_f32.mean_all()?;
        let output_min = output_f32.min_all()?;
        let output_max = output_f32.max_all()?;

        println!("  Output mean: {:.6}", output_mean.to_scalar::<f32>()?);
        println!("  Output min: {:.6}", output_min.to_scalar::<f32>()?);
        println!("  Output max: {:.6}", output_max.to_scalar::<f32>()?);
    }

    println!("\n🎉 Example completed successfully!");

    Ok(())
}
