//! cuDNN graph wrapper for attention operations
//!
//! This module provides a safe wrapper around cuDNN's graph API for building
//! and executing attention operations.

use crate::error::{CuDNNError, Result};
use crate::ffi;
use crate::tensor::{CuDNNTensor, DataType};
use std::ptr;

/// cuDNN graph for building operations
pub struct CuDNNGraph {
    handle: ffi::cudnnHandle_t,
    graph: ffi::cudnnBackendDescriptor_t,
    execution_plan: Option<ffi::cudnnBackendDescriptor_t>,
    workspace_size: usize,
}

impl CuDNNGraph {
    /// Create a new cuDNN graph
    pub fn new() -> Result<Self> {
        let mut handle: ffi::cudnnHandle_t = ptr::null_mut();
        let result = unsafe { ffi::cudnnCreate(&mut handle) };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to create cuDNN handle",
            ));
        }

        // Create a backend descriptor for the graph
        let mut graph: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
                &mut graph,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroy(handle) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to create graph descriptor",
            ));
        }

        Ok(Self {
            handle,
            graph,
            execution_plan: None,
            workspace_size: 0,
        })
    }

    /// Build SDPA operation graph for variable length attention
    pub fn build_sdpa_varlen(
        &mut self,
        q: &CuDNNTensor,
        k: &CuDNNTensor,
        v: &CuDNNTensor,
        ragged_offset_q: &CuDNNTensor,
        ragged_offset_k: &CuDNNTensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<CuDNNTensor> {
        // Get output dimensions from Q
        let o_dims = q.dims().to_vec();
        let data_type = q.data_type();

        // Create output tensor descriptor
        let o = CuDNNTensor::new(o_dims, data_type)?;

        // Create SDPA forward operation descriptor
        let mut sdpa_op: ffi::cudnnBackendDescriptor_t = ptr::null_mut();
        let result = unsafe {
            ffi::cudnnBackendCreateDescriptor(
                ffi::cudnnBackendDescriptorType_t_CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR,
                &mut sdpa_op,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to create SDPA operation descriptor",
            ));
        }

        // Set Q tensor
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &q.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set Q tensor in SDPA operation",
            ));
        }

        // Set K tensor
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &k.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set K tensor in SDPA operation",
            ));
        }

        // Set V tensor
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &v.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set V tensor in SDPA operation",
            ));
        }

        // Set O (output) tensor
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &o.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set O tensor in SDPA operation",
            ));
        }

        // Set ragged offset for Q (sequence lengths)
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &ragged_offset_q.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set Q ragged offset in SDPA operation",
            ));
        }

        // Set ragged offset for K/V (sequence lengths)
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &ragged_offset_k.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set K/V ragged offset in SDPA operation",
            ));
        }

        // Create scale tensor descriptor
        let scale_data_type = DataType::Float32; // Scale is always float32
        let scale_dims = vec![1_i64, 1, 1, 1];
        let scale_tensor = CuDNNTensor::new(scale_dims, scale_data_type)?;

        // Set scale
        let result = unsafe {
            ffi::cudnnBackendSetAttribute(
                sdpa_op,
                ffi::cudnnBackendAttributeName_t_CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                ffi::cudnnBackendAttributeType_t_CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &scale_tensor.descriptor() as *const _ as *const std::ffi::c_void,
            )
        };

        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to set scale in SDPA operation",
            ));
        }

        // Finalize the operation descriptor
        let result = unsafe { ffi::cudnnBackendFinalize(sdpa_op) };
        if result != ffi::cudnnStatus_t_CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };
            return Err(CuDNNError::cudnn_error(
                result as i32,
                "Failed to finalize SDPA operation",
            ));
        }

        // Store the operation in the graph
        // For now, we just store the operation descriptor
        // In a full implementation, we'd add it to an operation graph

        // Clean up
        unsafe { ffi::cudnnBackendDestroyDescriptor(sdpa_op) };

        // Return the output tensor descriptor
        Ok(o)
    }

    /// Build the graph and create an execution plan
    pub fn build(&mut self) -> Result<()> {
        // For now, return an error indicating this needs implementation
        Err(CuDNNError::not_available(
            "Graph building not yet fully implemented",
        ))
    }

    /// Execute the graph
    pub fn execute(&self, _workspace: &mut [u8]) -> Result<()> {
        if self.execution_plan.is_none() {
            return Err(CuDNNError::invalid_configuration("Graph not built"));
        }

        // For now, return an error indicating this needs implementation
        Err(CuDNNError::not_available(
            "Graph execution not yet fully implemented",
        ))
    }
}

impl Drop for CuDNNGraph {
    fn drop(&mut self) {
        unsafe {
            if let Some(plan) = self.execution_plan {
                ffi::cudnnBackendDestroyDescriptor(plan);
            }
            if !self.graph.is_null() {
                ffi::cudnnBackendDestroyDescriptor(self.graph);
            }
            if !self.handle.is_null() {
                ffi::cudnnDestroy(self.handle);
            }
        }
    }
}
