//! cuDNN tensor abstractions
//!
//! This module provides safe wrappers around cuDNN tensor descriptors.

use crate::error::{CuDNNError, Result};
use crate::ffi;
use std::ptr;

/// Data type for cuDNN tensors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int8,
}

impl DataType {
    /// Convert to cuDNN data type constant
    pub fn to_cudnn(&self) -> ffi::cudnnDataType_t {
        match self {
            DataType::Float32 => ffi::cudnnDataType_t_CUDNN_DATA_FLOAT,
            DataType::Float16 => ffi::cudnnDataType_t_CUDNN_DATA_HALF,
            DataType::BFloat16 => ffi::cudnnDataType_t_CUDNN_DATA_BFLOAT16,
            DataType::Int32 => ffi::cudnnDataType_t_CUDNN_DATA_INT32,
            DataType::Int8 => ffi::cudnnDataType_t_CUDNN_DATA_INT8,
        }
    }

    /// Convert from candle DType
    pub fn from_candle(dtype: candle::DType) -> Option<Self> {
        match dtype {
            candle::DType::F32 => Some(DataType::Float32),
            candle::DType::F16 => Some(DataType::Float16),
            candle::DType::BF16 => Some(DataType::BFloat16),
            _ => None,
        }
    }
}

/// cuDNN tensor descriptor wrapper
pub struct CuDNNTensor {
    pub(crate) desc: ffi::cudnnTensorDescriptor_t,
    pub(crate) dims: Vec<i32>,
    pub(crate) data_type: DataType,
}

impl CuDNNTensor {
    /// Create a new tensor descriptor
    pub fn new(dims: Vec<i32>, data_type: DataType) -> Result<Self> {
        let mut desc: ffi::cudnnTensorDescriptor_t = ptr::null_mut();

        let result = unsafe { ffi::cudnnCreateTensorDescriptor(&mut desc) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to create tensor descriptor",
            ));
        }

        // Set tensor descriptor
        let result = unsafe {
            ffi::cudnnSetTensorNdDescriptor(
                desc,
                data_type.to_cudnn(),
                dims.len() as i32,
                dims.as_ptr(),
                dims.as_ptr(), // strides - using same as dims for now
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroyTensorDescriptor(desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor descriptor",
            ));
        }

        Ok(Self {
            desc,
            dims,
            data_type,
        })
    }

    /// Get the underlying descriptor
    pub fn desc(&self) -> ffi::cudnnTensorDescriptor_t {
        self.desc
    }

    /// Get dimensions
    pub fn dims(&self) -> &[i32] {
        &self.dims
    }

    /// Get data type
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

impl Drop for CuDNNTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.desc.is_null() {
                ffi::cudnnDestroyTensorDescriptor(self.desc);
            }
        }
    }
}
