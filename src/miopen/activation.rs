// src/miopen/activation.rs

use crate::hip::DeviceMemory;
use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::c_void;
use std::ptr;

/// Activation mode type
#[repr(u32)]
pub enum ActivationMode {
    MiopenActivationPASTHRU = ffi::miopenActivationMode_t_miopenActivationPASTHRU,
    MiopenActivationLOGISTIC = ffi::miopenActivationMode_t_miopenActivationLOGISTIC,
    MiopenActivationTANH = ffi::miopenActivationMode_t_miopenActivationTANH,
    MiopenActivationRELU = ffi::miopenActivationMode_t_miopenActivationRELU,
    MiopenActivationSOFTRELU = ffi::miopenActivationMode_t_miopenActivationSOFTRELU,
    MiopenActivationABS = ffi::miopenActivationMode_t_miopenActivationABS,
    MiopenActivationPOWER = ffi::miopenActivationMode_t_miopenActivationPOWER,
    MiopenActivationCLIPPEDRELU = ffi::miopenActivationMode_t_miopenActivationCLIPPEDRELU,
    MiopenActivationLEAKYRELU = ffi::miopenActivationMode_t_miopenActivationLEAKYRELU,
    MiopenActivationELU = ffi::miopenActivationMode_t_miopenActivationELU,
}

impl TryFrom<u32> for ActivationMode {
    type Error = Error;

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            ffi::miopenActivationMode_t_miopenActivationPASTHRU => {
                Ok(ActivationMode::MiopenActivationPASTHRU)
            }
            ffi::miopenActivationMode_t_miopenActivationLOGISTIC => {
                Ok(ActivationMode::MiopenActivationLOGISTIC)
            }
            ffi::miopenActivationMode_t_miopenActivationTANH => {
                Ok(ActivationMode::MiopenActivationTANH)
            }
            ffi::miopenActivationMode_t_miopenActivationRELU => {
                Ok(ActivationMode::MiopenActivationRELU)
            }
            ffi::miopenActivationMode_t_miopenActivationSOFTRELU => {
                Ok(ActivationMode::MiopenActivationSOFTRELU)
            }
            ffi::miopenActivationMode_t_miopenActivationABS => {
                Ok(ActivationMode::MiopenActivationABS)
            }
            ffi::miopenActivationMode_t_miopenActivationPOWER => {
                Ok(ActivationMode::MiopenActivationPOWER)
            }
            ffi::miopenActivationMode_t_miopenActivationCLIPPEDRELU => {
                Ok(ActivationMode::MiopenActivationCLIPPEDRELU)
            }
            ffi::miopenActivationMode_t_miopenActivationLEAKYRELU => {
                Ok(ActivationMode::MiopenActivationLEAKYRELU)
            }
            ffi::miopenActivationMode_t_miopenActivationELU => {
                Ok(ActivationMode::MiopenActivationELU)
            }
            _ => Err(Error::new(ffi::miopenStatus_t_miopenStatusUnknownError)),
        }
    }
}

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

    pub fn with_mode(mode: ActivationMode, alpha: f64, beta: f64, gamma: f64) -> Result<Self> {
        let mut s = Self::new()?;
        s.set(mode, alpha, beta, gamma)?;
        Ok(s)
    }

    /// Set the activation descriptor details
    pub fn set(&mut self, mode: ActivationMode, alpha: f64, beta: f64, gamma: f64) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetActivationDescriptor(self.desc, mode as u32, alpha, beta, gamma)
        };

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

        Ok((ActivationMode::try_from(mode)?, alpha, beta, gamma))
    }

    /// Execute a forward activation operation
    pub fn forward<T>(
        &self,
        handle: &Handle,
        alpha: &f32,
        x_desc: &TensorDescriptor,
        x: &DeviceMemory<T>,
        beta: &f32,
        y_desc: &TensorDescriptor,
        y: &mut DeviceMemory<T>,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenActivationForward(
                handle.as_raw(),
                self.desc,
                alpha as *const _ as *const c_void,
                x_desc.as_raw(),
                x.as_ptr(),
                beta as *const _ as *const c_void,
                y_desc.as_raw(),
                y.as_ptr(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute a backward activation operation
    pub unsafe fn backward<T>(
        &self,
        handle: &Handle,
        alpha: &f32,
        y_desc: &TensorDescriptor,
        y: &DeviceMemory<T>,
        dy_desc: &TensorDescriptor,
        dy: &DeviceMemory<T>,
        x_desc: &TensorDescriptor,
        x: &DeviceMemory<T>,
        beta: &f32,
        dx_desc: &TensorDescriptor,
        dx: &mut DeviceMemory<T>,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenActivationBackward(
                handle.as_raw(),
                self.desc,
                alpha as *const _ as *const c_void,
                y_desc.as_raw(),
                y.as_ptr(),
                dy_desc.as_raw(),
                dy.as_ptr(),
                x_desc.as_raw(),
                x.as_ptr(),
                beta as *const _ as *const c_void,
                dx_desc.as_raw(),
                dx.as_ptr(),
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
