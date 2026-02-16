//! cuDNN attention bindings for candle
//!
//! This crate provides high-performance attention operations using NVIDIA's cuDNN frontend API.
//! It supports variable length sequences through ragged layout and offers significant performance
//! improvements over custom CUDA implementations.
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
//! use candle_cudnn_attn::cudnn_attention_varlen;
//!
//! let q = Tensor::randn(0.0, 1.0, (2, 8, 128, 64), &candle::Device::new_cuda(0)?)?;
//! let k = Tensor::randn(0.0, 1.0, (2, 8, 128, 64), &candle::Device::new_cuda(0)?)?;
//! let v = Tensor::randn(0.0, 1.0, (2, 8, 128, 64), &candle::Device::new_cuda(0)?)?;
//!
//! // Ragged offset for variable length sequences
//! let ragged_offset = Tensor::from_vec(vec![0u32, 128, 256], (3, 1, 1, 1), &candle::Device::new_cuda(0)?)?;
//!
//! let output = cudnn_attention_varlen(
//!     &q, &k, &v, &ragged_offset, 128, 1.0 / (64.0_f32).sqrt(), true
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
    // Check CUDA availability and cuDNN version
    // This will be implemented when we have the FFI bindings
    false // Placeholder
}

/// Get cuDNN version information
pub fn version_info() -> Result<String> {
    // Return cuDNN version
    // This will be implemented when we have the FFI bindings
    Ok("cuDNN version not yet implemented".to_string())
}
