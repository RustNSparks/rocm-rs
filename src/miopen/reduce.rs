// src/miopen/reduce.rs

use std::ptr;
use std::os::raw::c_void;
use crate::miopen::ffi;
use crate::miopen::error::{Error, Result};
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;

/// Reduction tensor operation
pub type ReduceTensorOp = ffi::miopenReduceTensorOp_t;

/// NaN propagation mode
pub type NanPropagation = ffi::miopenNanPropagation_t;

/// Reduction tensor indices
pub type ReduceTensorIndices = ffi::miopenReduceTensorIndices_t;

/// Indices type
pub type IndicesType = ffi::miopenIndicesType_t;

/// Safe wrapper for MIOpen reduce tensor descriptor
pub struct ReduceTensorDescriptor {
    desc: ffi::miopenReduceTensorDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for ReduceTensorDescriptor {}
unsafe impl Sync for ReduceTensorDescriptor {}

impl ReduceTensorDescriptor {
    /// Create a new reduce tensor descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateReduceTensorDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the reduce tensor descriptor
    pub fn set(&mut self,
               reduce_op: ReduceTensorOp,
               comp_type: ffi::miopenDataType_t,
               nan_opt: NanPropagation,
               indices: ReduceTensorIndices,
               indices_type: IndicesType) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetReduceTensorDescriptor(
                self.desc,
                reduce_op,
                comp_type,
                nan_opt,
                indices,
                indices_type,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the reduce tensor descriptor details
    pub fn get(&self) -> Result<(ReduceTensorOp, ffi::miopenDataType_t, NanPropagation, ReduceTensorIndices, IndicesType)> {
        let mut reduce_op = 0;
        let mut comp_type = 0;
        let mut nan_opt = 0;
        let mut indices = 0;
        let mut indices_type = 0;

        let status = unsafe {
            ffi::miopenGetReduceTensorDescriptor(
                self.desc,
                &mut reduce_op,
                &mut comp_type,
                &mut nan_opt,
                &mut indices,
                &mut indices_type,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((reduce_op, comp_type, nan_opt, indices, indices_type))
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenReduceTensorDescriptor_t {
        self.desc
    }
}

impl Drop for ReduceTensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyReduceTensorDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

/// Get the size required for reduction indices
pub fn get_reduction_indices_size(handle: &Handle, reduce_desc: &ReduceTensorDescriptor,
                                  a_desc: &TensorDescriptor, c_desc: &TensorDescriptor) -> Result<usize> {
    let mut size_in_bytes = 0;

    let status = unsafe {
        ffi::miopenGetReductionIndicesSize(
            handle.as_raw(),
            reduce_desc.as_raw(),
            a_desc.as_raw(),
            c_desc.as_raw(),
            &mut size_in_bytes,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(size_in_bytes)
}

/// Get the workspace size required for reduction
pub fn get_reduction_workspace_size(handle: &Handle, reduce_desc: &ReduceTensorDescriptor,
                                    a_desc: &TensorDescriptor, c_desc: &TensorDescriptor) -> Result<usize> {
    let mut size_in_bytes = 0;

    let status = unsafe {
        ffi::miopenGetReductionWorkspaceSize(
            handle.as_raw(),
            reduce_desc.as_raw(),
            a_desc.as_raw(),
            c_desc.as_raw(),
            &mut size_in_bytes,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(size_in_bytes)
}

/// Execute a reduction operation
pub fn reduce_tensor(handle: &Handle, reduce_desc: &ReduceTensorDescriptor,
                     indices: *mut c_void, indices_size: usize,
                     workspace: *mut c_void, workspace_size: usize,
                     alpha: &[u8], a_desc: &TensorDescriptor, a: *const c_void,
                     beta: &[u8], c_desc: &TensorDescriptor, c: *mut c_void) -> Result<()> {
    let status = unsafe {
        ffi::miopenReduceTensor(
            handle.as_raw(),
            reduce_desc.as_raw(),
            indices,
            indices_size,
            workspace,
            workspace_size,
            alpha.as_ptr() as *const c_void,
            a_desc.as_raw(),
            a,
            beta.as_ptr() as *const c_void,
            c_desc.as_raw(),
            c,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}