// src/miopen/softmax.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::c_void;
use std::ptr;

/// Softmax algorithm type
pub type SoftmaxAlgorithm = ffi::miopenSoftmaxAlgorithm_t;

/// Constants for softmax algorithms
pub mod softmax_algorithm {
    use crate::miopen::ffi;

    /// Fast softmax implementation
    pub const FAST: super::SoftmaxAlgorithm = ffi::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_FAST;

    /// Accurate softmax implementation
    pub const ACCURATE: super::SoftmaxAlgorithm =
        ffi::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_ACCURATE;

    /// Log softmax implementation
    pub const LOG: super::SoftmaxAlgorithm = ffi::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_LOG;
}

/// Softmax mode type
pub type SoftmaxMode = ffi::miopenSoftmaxMode_t;

/// Constants for softmax modes
pub mod softmax_mode {
    use crate::miopen::ffi;

    /// Per instance softmax
    pub const INSTANCE: super::SoftmaxMode = ffi::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_INSTANCE;

    /// Per channel softmax
    pub const CHANNEL: super::SoftmaxMode = ffi::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_CHANNEL;
}

/// Safe wrapper for MIOpen Softmax descriptor
pub struct SoftmaxDescriptor {
    desc: ffi::miopenSoftmaxDescriptor_t,
}

impl SoftmaxDescriptor {
    /// Create a new softmax descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateSoftmaxDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the softmax descriptor parameters
    pub fn set(
        &mut self,
        alpha: f32,
        beta: f32,
        algorithm: SoftmaxAlgorithm,
        mode: SoftmaxMode,
    ) -> Result<()> {
        let status =
            unsafe { ffi::miopenSetSoftmaxDescriptor(self.desc, alpha, beta, algorithm, mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the softmax descriptor parameters
    pub fn get(&self) -> Result<(f32, f32, SoftmaxAlgorithm, SoftmaxMode)> {
        let mut alpha = 0.0f32;
        let mut beta = 0.0f32;
        let mut algorithm = 0;
        let mut mode = 0;

        let status = unsafe {
            ffi::miopenGetSoftmaxDescriptor(
                self.desc,
                &mut alpha,
                &mut beta,
                &mut algorithm,
                &mut mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((alpha, beta, algorithm, mode))
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenSoftmaxDescriptor_t {
        self.desc
    }
}

/// Execute a softmax forward operation
pub unsafe fn softmax_forward(
    handle: &Handle,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenSoftmaxForward(
            handle.as_raw(),
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

/// Execute a softmax forward operation with specified algorithm and mode
pub unsafe fn softmax_forward_v2(
    handle: &Handle,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenSoftmaxForward_V2(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            x_desc.as_raw(),
            x,
            beta.as_ptr() as *const c_void,
            y_desc.as_raw(),
            y,
            algorithm,
            mode,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute a softmax backward operation
pub unsafe fn softmax_backward(
    handle: &Handle,
    alpha: &[u8],
    y_desc: &TensorDescriptor,
    y: *const c_void,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    beta: &[u8],
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenSoftmaxBackward(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            y_desc.as_raw(),
            y,
            dy_desc.as_raw(),
            dy,
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

/// Execute a softmax backward operation with specified algorithm and mode
pub unsafe fn softmax_backward_v2(
    handle: &Handle,
    alpha: &[u8],
    y_desc: &TensorDescriptor,
    y: *const c_void,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    beta: &[u8],
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
    algorithm: SoftmaxAlgorithm,
    mode: SoftmaxMode,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenSoftmaxBackward_V2(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            y_desc.as_raw(),
            y,
            dy_desc.as_raw(),
            dy,
            beta.as_ptr() as *const c_void,
            dx_desc.as_raw(),
            dx,
            algorithm,
            mode,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}
