//! cuDNN tensor abstractions
//!
//! This module provides safe wrappers around cuDNN tensor descriptors.

use crate::error::{CuDNNError, Result};
use crate::ffi;
use std::ptr;
use std::sync::atomic::{AtomicI64, Ordering};

static NEXT_TENSOR_UID: AtomicI64 = AtomicI64::new(1);

/// Data type for cuDNN tensors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
    Int64,
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
            DataType::Int64 => ffi::cudnnDataType_t_CUDNN_DATA_INT64,
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
    pub(crate) dims: Vec<i64>,
    pub(crate) data_type: DataType,
    pub(crate) uid: i64,
}

impl CuDNNTensor {
    /// Create a new backend tensor descriptor for graph API
    pub fn new(dims: Vec<i64>, data_type: DataType) -> Result<Self> {
        let uid = NEXT_TENSOR_UID.fetch_add(1, Ordering::Relaxed);
        let mut strides = vec![0i64; dims.len()];
        let mut stride = 1i64;
        for i in (0..dims.len()).rev() {
            strides[i] = stride;
            stride *= dims[i];
        }
        Self::new_with_uid_and_strides(dims, strides, data_type, uid)
    }

    /// Create a tensor descriptor with explicit UID and strides.
    pub fn new_with_uid_and_strides(
        dims: Vec<i64>,
        strides: Vec<i64>,
        data_type: DataType,
        uid: i64,
    ) -> Result<Self> {
        Self::new_with_uid_and_strides_and_ragged(dims, strides, data_type, uid, None)
    }

    /// Create a tensor descriptor with optional ragged-offset descriptor.
    pub fn new_with_uid_and_strides_and_ragged(
        dims: Vec<i64>,
        strides: Vec<i64>,
        data_type: DataType,
        uid: i64,
        ragged_offset_desc: Option<ffi::cudnnBackendDescriptor_t>,
    ) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(CuDNNError::invalid_configuration(format!(
                "Tensor dims and strides rank mismatch: {} != {}",
                dims.len(),
                strides.len()
            )));
        }

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

        // Set dimensions.
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

        // Set strides.
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

        // Set unique ID so variant-pack can bind pointers to tensors.
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                backend_desc,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_UNIQUE_ID,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                1,
                &uid as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor unique ID",
            ));
        }

        // cuDNN backend descriptors require explicit byte alignment.
        let alignment: i64 = 16;
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                backend_desc,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_INT64,
                1,
                &alignment as *const _ as *const std::ffi::c_void,
            )
        };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set tensor byte alignment",
            ));
        }

        if let Some(ragged_desc) = ragged_offset_desc {
            let result = unsafe {
                ffi::cudnnBackendSetAttribute(
                    backend_desc,
                    ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                    ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                    1,
                    &ragged_desc as *const _ as *const std::ffi::c_void,
                )
            };
            if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
                unsafe { ffi::cudnnBackendDestroyDescriptor(backend_desc) };
                return Err(CuDNNError::cudnn_error(
                    result as i32,
                    "Failed to set tensor ragged offset descriptor",
                ));
            }
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
            uid,
        })
    }

    /// Get the underlying backend descriptor
    pub fn descriptor(&self) -> ffi::cudnnBackendDescriptor_t {
        self.backend_desc
    }

    /// Get dimensions
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    /// Get data type
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Get stable tensor UID used by variant packs.
    pub fn uid(&self) -> i64 {
        self.uid
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
