# candle-cudnn-attn

Flash-attention compatible cuDNN attention operations for Candle.

## Overview

This crate provides high-performance attention operations using NVIDIA's cuDNN frontend API with a flash-attention-v1/v3 compatible API.

## Features

- **3D Tensor API**: `(num_heads, total_tokens, head_dim)` - no padding required
- **Variable Length Sequences**: Support for different sequence lengths in the same batch via cumulative sequence lengths (`cu_seqlens`)
- **Flash-Attention Compatible**: Drop-in replacement for flash-attention-v1/v3
- **Causal Masking**: Optional causal attention

## Requirements

- CUDA 12.0 or newer
- cuDNN 9.0 or newer
- GPU with SM80 (Ampere) architecture or newer

## Usage

```rust
use candle::{Device, Tensor};
use candle_cudnn_attn::flash_attn_varlen;

let device = Device::new_cuda(0)?;

// 3D tensors: (num_heads, total_tokens, head_dim)
let num_heads = 8;
let total_tokens = 256; // e.g., batch_size=2, seq_len=128 each
let head_dim = 64;

let q = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;
let k = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;
let v = Tensor::randn(0.0f32, 1.0f32, (num_heads, total_tokens, head_dim), &device)?;

// Cumulative sequence lengths: (batch_size + 1,)
// For batch_size=2 with seq_len=128 each: [0, 128, 256]
let seqlens_q = Tensor::new(&[0u32, 128u32, 256u32], &device)?;
let seqlens_k = Tensor::new(&[0u32, 128u32, 256u32], &device)?;

let output = flash_attn_varlen(
    &q, &k, &v,
    &seqlens_q, &seqlens_k,
    128, 128,  // max_seqlen_q, max_seqlen_k
    1.0 / (head_dim as f32).sqrt(),  // softmax_scale
    true,  // causal
)?;

// Output shape: (num_heads, total_q, head_dim)
```

## API

### `flash_attn_varlen`

Main function for variable length attention with flash-attention compatible API.

**Arguments:**
- `q`: Query tensor `(num_heads, total_q, head_dim)`
- `k`: Key tensor `(num_heads, total_k, head_dim)`
- `v`: Value tensor `(num_heads, total_k, head_dim)`
- `seqlens_q`: Cumulative sequence lengths for queries `(batch_size + 1,)`
- `seqlens_k`: Cumulative sequence lengths for keys/values `(batch_size + 1,)`
- `max_seqlen_q`: Maximum query sequence length
- `max_seqlen_k`: Maximum key/value sequence length
- `softmax_scale`: Scale factor for attention scores
- `causal`: Whether to apply causal masking

**Returns:**
Output tensor `(num_heads, total_q, head_dim)`

## Testing

Run tests with:
```bash
cargo test --package candle-cudnn-attn
```

## License

Apache-2.0 or MIT
