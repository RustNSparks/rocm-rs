// src/miopen/lrn.rs

use std::ptr;
use std::os::raw::c_void;
use crate::miopen::ffi;
use crate::miopen::error::{Error, Result};
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;

/// LRN mode type
pub type LRNMode = ffi::miopenLRNMode_t;

/// Safe wrapper for MIOpen LRN descriptor
pub struct LRNDescriptor {
    desc: ffi::miopenLRNDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for LRNDescriptor {}
unsafe impl Sync for LRNDescriptor {}

impl LRNDescriptor {
    /// Create a new LRN descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateLRNDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the LRN descriptor details
    pub fn set(&mut self, mode: LRNMode, lrn_n: u32, lrn_alpha: f64, lrn_beta: f64, lrn_k: f64) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetLRNDescriptor(self.desc, mode, lrn_n, lrn_alpha, lrn_beta, lrn_k)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the LRN descriptor details
    pub fn get(&self) -> Result<(LRNMode, u32, f64, f64, f64)> {
        let mut mode = 0;
        let mut lrn_n = 0;
        let mut lrn_alpha = 0.0;
        let mut lrn_beta = 0.0;
        let mut lrn_k = 0.0;

        let status = unsafe {
            ffi::miopenGetLRNDescriptor(self.desc, &mut mode, &mut lrn_n, &mut lrn_alpha, &mut lrn_beta, &mut lrn_k)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((mode, lrn_n, lrn_alpha, lrn_beta, lrn_k))
    }

    /// Get the workspace size required for LRN operations
    pub fn get_workspace_size(y_desc: &TensorDescriptor) -> Result<usize> {
        let mut workspace_size = 0;

        let status = unsafe {
            ffi::miopenLRNGetWorkSpaceSize(y_desc.as_raw(), &mut workspace_size)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(workspace_size)
    }

    /// Execute a forward LRN operation
    pub fn forward(
        &self,
        handle: &Handle,
        alpha: &[u8],
        x_desc: &TensorDescriptor,
        x: *const c_void,
        beta: &[u8],
        y_desc: &TensorDescriptor,
        y: *mut c_void,
        do_backward: bool,
        workspace: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenLRNForward(
                handle.as_raw(),
                self.desc,
                alpha.as_ptr() as *const c_void,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const c_void,
                y_desc.as_raw(),
                y,
                do_backward,
                workspace,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute a backward LRN operation
    pub fn backward(
        &self,
        handle: &Handle,
        alpha: &[u8],
        y_desc: &TensorDescriptor,
        y: *const c_void,
        dy_desc: &TensorDescriptor,
        dy: *const c_void,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        beta: &[u8],
        dx_desc: &TensorDescriptor,
        dx: *mut c_void,
        workspace: *const c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenLRNBackward(
                handle.as_raw(),
                self.desc,
                alpha.as_ptr() as *const c_void,
                y_desc.as_raw(),
                y,
                dy_desc.as_raw(),
                dy,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const c_void,
                dx_desc.as_raw(),
                dx,
                workspace,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenLRNDescriptor_t {
        self.desc
    }
}

impl Drop for LRNDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyLRNDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}