//! Basic cuDNN attention example

use candle::{Device, Tensor};
use candle_cudnn_attn::cudnn_attention_varlen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cuDNN Attention Example");
    println!("======================");

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

    // Create test tensors
    let batch_size = 4;
    let num_heads = 8;
    let seq_len = 128;
    let head_dim = 64;

    println!("\n📊 Tensor Configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Number of heads: {}", num_heads);
    println!("  Sequence length: {}", seq_len);
    println!("  Head dimension: {}", head_dim);

    // Create random tensors
    let q = Tensor::randn(
        0.0,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0.0,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0.0,
        1.0,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    println!("✅ Created input tensors");

    // Create ragged offset for variable length sequences
    // In this example, we'll use uniform lengths for simplicity
    let mut offsets = vec![0];
    for i in 1..=batch_size {
        offsets.push((i * seq_len * num_heads * head_dim) as i32);
    }

    let ragged_offset = Tensor::from_vec(
        offsets.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        (batch_size + 1, 1, 1, 1),
        &device,
    )?;
    println!("✅ Created ragged offset tensor");

    // Run attention
    println!("\n🚀 Running cuDNN attention...");
    let start = std::time::Instant::now();

    let output = cudnn_attention_varlen(
        &q,
        &k,
        &v,
        &ragged_offset,
        seq_len,
        1.0 / (head_dim as f32).sqrt(),
        true,
    )?;

    let duration = start.elapsed();
    println!("✅ Attention completed in {:?}", duration);

    // Verify output
    println!("\n📋 Output Verification:");
    println!("  Input shape: {:?}", q.shape());
    println!("  Output shape: {:?}", output.shape());

    // Check some basic statistics
    let output_mean = output.mean_all()?;
    let output_min = output.min_all()?;
    let output_max = output.max_all()?;

    println!("  Output mean: {:.6}", output_mean.to_scalar::<f32>()?);
    println!("  Output min: {:.6}", output_min.to_scalar::<f32>()?);
    println!("  Output max: {:.6}", output_max.to_scalar::<f32>()?);

    println!("\n🎉 Example completed successfully!");

    Ok(())
}
