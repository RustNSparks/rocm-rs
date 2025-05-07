// src/miopen/activation.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::c_void;
use std::ptr;

/// Activation mode type
pub type ActivationMode = ffi::miopenActivationMode_t;

/// Safe wrapper for MIOpen activation descriptor
pub struct ActivationDescriptor {
    desc: ffi::miopenActivationDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for ActivationDescriptor {}
unsafe impl Sync for ActivationDescriptor {}

impl ActivationDescriptor {
    /// Create a new activation descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateActivationDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the activation descriptor details
    pub fn set(&mut self, mode: ActivationMode, alpha: f64, beta: f64, gamma: f64) -> Result<()> {
        let status =
            unsafe { ffi::miopenSetActivationDescriptor(self.desc, mode, alpha, beta, gamma) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the activation descriptor details
    pub fn get(&self) -> Result<(ActivationMode, f64, f64, f64)> {
        let mut mode = 0;
        let mut alpha = 0.0;
        let mut beta = 0.0;
        let mut gamma = 0.0;

        let status = unsafe {
            ffi::miopenGetActivationDescriptor(
                self.desc, &mut mode, &mut alpha, &mut beta, &mut gamma,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((mode, alpha, beta, gamma))
    }

    /// Execute a forward activation operation
    pub unsafe fn forward(
        &self,
        handle: &Handle,
        alpha: &[u8],
        x_desc: &TensorDescriptor,
        x: *const c_void,
        beta: &[u8],
        y_desc: &TensorDescriptor,
        y: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenActivationForward(
                handle.as_raw(),
                self.desc,
                alpha.as_ptr() as *const c_void,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const c_void,
                y_desc.as_raw(),
                y,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute a backward activation operation
    pub unsafe fn backward(
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
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenActivationBackward(
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
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenActivationDescriptor_t {
        self.desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyActivationDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}
