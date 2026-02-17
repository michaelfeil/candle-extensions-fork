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

## Installing cuDNN

### Option 1: Using pip (Recommended for Python environments)

```bash
pip install nvidia-cudnn-cu12
```

This will install cuDNN 9.x for CUDA 12.x in your Python environment.

### Option 2: Using NVIDIA's apt repository (System-wide installation)

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install cuDNN
sudo apt-get install libcudnn9-cuda-12
sudo apt-get install libcudnn9-dev-cuda-12  # For development headers
```

### Option 3: Manual download from NVIDIA

1. Go to https://developer.nvidia.com/cudnn-downloads
2. Download the cuDNN 9.x for CUDA 12.x tarball
3. Extract and copy files to CUDA installation:

```bash
tar -xvf cudnn-linux-x86_64-9.x.x.x_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-9.x.x.x_cuda12-archive/include/cudnn*.h /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-9.x.x.x_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Verifying Installation

```bash
# Check cuDNN version
python -c "import nvidia.cudnn; print(nvidia.cudnn.__version__)" 2>/dev/null || echo "Not installed via pip"

# Or check system installation
ls -la /usr/local/cuda/lib64/libcudnn* 2>/dev/null || echo "Not in CUDA directory"
ls -la /usr/lib/x86_64-linux-gnu/libcudnn* 2>/dev/null || echo "Not in system lib"
```

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
