//! cuDNN attention bindings for candle
//!
//! This crate provides high-performance attention operations using NVIDIA's cuDNN frontend API.
//! It supports variable length sequences through cumulative sequence lengths (cu_seqlens)
//! and offers significant performance improvements over custom CUDA implementations.
//!
//! This crate follows the flash-attention-v1/v3 API for easy integration.

pub mod attention;
pub mod error;
pub mod ffi;
pub mod frontend;
pub mod graph;
pub mod tensor;

pub use attention::*;
pub use error::*;
pub use graph::*;
pub use tensor::*;

pub fn is_available() -> bool {
    attention::is_available()
}

pub fn version_info() -> Result<String> {
    attention::version_info()
}
