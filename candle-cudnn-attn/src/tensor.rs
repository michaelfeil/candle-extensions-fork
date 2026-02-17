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

/// cuDNN backend tensor descriptor wrapper for graph API
pub struct CuDNNTensor {
    pub(crate) backend_desc: ffi::cudnnBackendDescriptor_t,
    pub(crate) dims: Vec<i32>,
    pub(crate) data_type: DataType,
}

impl CuDNNTensor {
    /// Create a new backend tensor descriptor for graph API
    pub fn new(dims: Vec<i32>, data_type: DataType) -> Result<Self> {
        let mut backend_desc: ffi::cudnnBackendDescriptor_t = ptr::null_mut();

        // Create backend tensor descriptor
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_TENSOR_DESCRIPTOR,
                &mut backend_desc,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to create backend tensor descriptor",
            ));
        }

        // Set data type
        let cudnn_dtype = data_type.to_cudnn();
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                backend_desc,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_DATA_TYPE,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_DATA_TYPE,
                1,
                &cudnn_dtype as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor data type",
            ));
        }

        // Set dimensions
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                backend_desc,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_DIMENSIONS,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                dims.len() as i64,
                dims.as_ptr() as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor dimensions",
            ));
        }

        // Set strides (row-major layout)
        let mut strides = vec![0i64; dims.len()];
        let mut stride = 1i64;
        for i in (0..dims.len()).rev() {
            strides[i] = stride;
            stride *= dims[i] as i64;
        }

        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                backend_desc,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_STRIDES,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                strides.len() as i64,
                strides.as_ptr() as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor strides",
            ));
        }

        // Finalize the descriptor
        let result = unsafe { ffi::cudnnBackendFinalize(backend_desc) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to finalize tensor descriptor",
            ));
        }

        Ok(Self {
            backend_desc,
            dims,
            data_type,
        })
    }

    /// Get the underlying backend descriptor
    pub fn descriptor(&self) -> ffi::cudnnBackendDescriptor_t {
        self.backend_desc
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
            if !self.backend_desc.is_null() {
                ffi::cudnnBackendDestroyDescriptor(self.backend_desc);
            }
        }
    }
}
