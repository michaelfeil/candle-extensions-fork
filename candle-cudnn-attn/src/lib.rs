//! cuDNN attention bindings for candle
//!
//! This crate provides high-performance attention operations using NVIDIA's cuDNN frontend API.
//! It supports variable length sequences through cumulative sequence lengths (cu_seqlens)
//! and offers significant performance improvements over custom CUDA implementations.
//!
//! This crate follows the flash-attention-v1/v3 API for easy integration.
//!
//! # Requirements
//!
//! - CUDA 12.0 or newer
//! - cuDNN 9.0 or newer  
//! - GPU with SM80 (Ampere) architecture or newer
//!
//! # Example
//!
//! ```rust,ignore
//! use candle::Tensor;
//! use candle_cudnn_attn::flash_attn_varlen;
//!
//! // Create 3D tensors: (num_heads, total_tokens, head_dim)
//! let q = Tensor::randn(0.0, 1.0, (8, 256, 64), &candle::Device::new_cuda(0)?)?;
//! let k = Tensor::randn(0.0, 1.0, (8, 256, 64), &candle::Device::new_cuda(0)?)?;
//! let v = Tensor::randn(0.0, 1.0, (8, 256, 64), &candle::Device::new_cuda(0)?)?;
//!
//! // Cumulative sequence lengths for batch_size=2 with sequences [128, 128]
//! let seqlens_q = Tensor::new(&[0u32, 128u32, 256u32], &candle::Device::new_cuda(0)?)?;
//! let seqlens_k = Tensor::new(&[0u32, 128u32, 256u32], &candle::Device::new_cuda(0)?)?;
//!
//! let output = flash_attn_varlen(
//!     &q, &k, &v,
//!     &seqlens_q, &seqlens_k,
//!     128, 128,  // max_seqlen_q, max_seqlen_k
//!     1.0 / (64.0_f32).sqrt(),  // softmax_scale
//!     true  // causal
//! )?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod attention;
pub mod error;
pub mod ffi;
pub mod graph;
pub mod tensor;

pub use attention::*;
pub use error::*;
pub use graph::*;
pub use tensor::*;

/// Check if cuDNN attention is available on the current system
pub fn is_available() -> bool {
    attention::is_available()
}

/// Get cuDNN version information
pub fn version_info() -> Result<String> {
    attention::version_info()
}
