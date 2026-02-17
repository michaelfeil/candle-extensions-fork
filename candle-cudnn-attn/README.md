# candle-cudnn-attn

cuDNN SDPA attention for Candle with THD varlen layout.

## Status

- THD layout supported: `(total_tokens, num_heads, head_dim)`
- Varlen supported via cumulative sequence lengths (`cu_seqlens`)
- `causal = false` and `causal = true` both supported
- Forward pass implemented

## Requirements

- CUDA 12+
- cuDNN 9.19+ (tested)
- GPU with modern cuDNN SDPA support (Ampere+ recommended)

This crate links cuDNN dynamically (`dylib`).

## API

```rust
pub fn flash_attn_varlen(
    q: &Tensor,            // (total_tokens, heads, dim)
    k: &Tensor,            // (total_tokens, heads, dim)
    v: &Tensor,            // (total_tokens, heads, dim)
    seqlens: &Tensor,      // U32, shape (batch + 1,), starts at 0, ends at total_tokens
    max_seqlen: usize,
    softmax_scale: f32,
    causal: bool,
) -> candle::Result<Tensor>
```

Output shape is `(total_tokens, heads, dim)`.

## Minimal Usage

```rust
use candle::{DType, Device, Tensor};
use candle_cudnn_attn::flash_attn_varlen;

let device = Device::new_cuda(0)?;
let (total_tokens, heads, dim) = (512, 8, 64);

let q = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, dim), &device)?.to_dtype(DType::F16)?;
let k = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, dim), &device)?.to_dtype(DType::F16)?;
let v = Tensor::randn(0.0f32, 1.0f32, (total_tokens, heads, dim), &device)?.to_dtype(DType::F16)?;

// Example: 4 sequences of length 128 -> [0, 128, 256, 384, 512]
let seqlens = Tensor::new(&[0u32, 128, 256, 384, 512], &device)?;

let out = flash_attn_varlen(
    &q,
    &k,
    &v,
    &seqlens,
    128,
    1.0 / (dim as f32).sqrt(),
    true,
)?;
```

## Build Notes

`build.rs` auto-detects headers/libs in common system paths. If needed, set:

- `CUDA_INCLUDE_DIR`
- `CUDNN_INCLUDE_DIR`
- `CUDNN_LIB_DIR`
- `CUDNN_FRONTEND_INCLUDE_DIR`

## Validation

- Example: `cargo run --example basic_attention`
- Tests: `cargo test --package candle-cudnn-attn`

## License

Apache-2.0 OR MIT
