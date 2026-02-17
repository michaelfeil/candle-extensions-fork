//! Error types for cuDNN attention operations

use candle::Error as CandleError;
use thiserror::Error;

/// Errors that can occur during cuDNN attention operations
#[derive(Error, Debug)]
pub enum CuDNNError {
    /// cuDNN frontend API error
    #[error("cuDNN frontend error: {message} (code: {code})")]
    FrontendError { code: i32, message: String },

    /// CUDA runtime error
    #[error("CUDA runtime error: {message} (code: {code})")]
    CudaError { code: i32, message: String },

    /// Invalid tensor shape or dimensions
    #[error("Invalid tensor shape: {expected}, got: {actual}")]
    InvalidShape { expected: String, actual: String },

    /// Unsupported data type
    #[error("Unsupported data type: {data_type}")]
    UnsupportedDataType { data_type: String },

    /// GPU compute capability not supported
    #[error("GPU compute capability {capability} not supported (requires SM80+)")]
    UnsupportedGPU { capability: String },

    /// Insufficient workspace memory
    #[error(
        "Insufficient workspace memory: required {required} bytes, available {available} bytes"
    )]
    InsufficientMemory { required: usize, available: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    /// cuDNN not available
    #[error("cuDNN not available: {reason}")]
    NotAvailable { reason: String },

    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl CuDNNError {
    /// Create a cuDNN error from status code
    pub fn from_cudnn_status(status: i32, message: impl Into<String>) -> Self {
        Self::FrontendError {
            code: status,
            message: message.into(),
        }
    }

    /// Create a new frontend error
    pub fn frontend(code: i32, message: impl Into<String>) -> Self {
        Self::FrontendError {
            code,
            message: message.into(),
        }
    }

    /// Create a new CUDA error
    pub fn cuda(code: i32, message: impl Into<String>) -> Self {
        Self::CudaError {
            code,
            message: message.into(),
        }
    }

    /// Create a new invalid shape error
    pub fn invalid_shape(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::InvalidShape {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a new unsupported data type error
    pub fn unsupported_data_type(data_type: impl Into<String>) -> Self {
        Self::UnsupportedDataType {
            data_type: data_type.into(),
        }
    }

    /// Create a new unsupported GPU error
    pub fn unsupported_gpu(capability: impl Into<String>) -> Self {
        Self::UnsupportedGPU {
            capability: capability.into(),
        }
    }

    /// Create a new insufficient memory error
    pub fn insufficient_memory(required: usize, available: usize) -> Self {
        Self::InsufficientMemory {
            required,
            available,
        }
    }

    /// Create a new invalid configuration error
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create a new not available error
    pub fn not_available(reason: impl Into<String>) -> Self {
        Self::NotAvailable {
            reason: reason.into(),
        }
    }

    /// Create a new internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, CuDNNError>;

impl From<CuDNNError> for CandleError {
    fn from(err: CuDNNError) -> Self {
        CandleError::msg(err.to_string())
    }
}

impl From<CandleError> for CuDNNError {
    fn from(err: CandleError) -> Self {
        CuDNNError::Internal {
            message: err.to_string(),
        }
    }
}
