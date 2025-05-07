// src/miopen/dropout.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::{c_ulonglong, c_void};
use std::ptr;

/// RNG type for dropout operations
pub type RNGType = ffi::miopenRNGType_t;

/// Safe wrapper for MIOpen dropout descriptor
pub struct DropoutDescriptor {
    pub(crate) desc: ffi::miopenDropoutDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for DropoutDescriptor {}
unsafe impl Sync for DropoutDescriptor {}

impl DropoutDescriptor {
    /// Create a new dropout descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateDropoutDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Initialize the dropout descriptor
    pub unsafe fn set(
        &mut self,
        handle: &Handle,
        dropout: f32,
        states: *mut c_void,
        state_size_in_bytes: usize,
        seed: c_ulonglong,
        use_mask: bool,
        state_evo: bool,
        rng_mode: RNGType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetDropoutDescriptor(
                self.desc,
                handle.as_raw(),
                dropout,
                states,
                state_size_in_bytes,
                seed,
                use_mask,
                state_evo,
                rng_mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Restore the dropout descriptor from a saved state
    pub unsafe fn restore(
        &mut self,
        handle: &Handle,
        dropout: f32,
        states: *mut c_void,
        state_size_in_bytes: usize,
        seed: c_ulonglong,
        use_mask: bool,
        state_evo: bool,
        rng_mode: RNGType,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenRestoreDropoutDescriptor(
                self.desc,
                handle.as_raw(),
                dropout,
                states,
                state_size_in_bytes,
                seed,
                use_mask,
                state_evo,
                rng_mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get dropout descriptor details
    pub fn get(
        &self,
        handle: &Handle,
    ) -> Result<(f32, *mut c_void, c_ulonglong, bool, bool, RNGType)> {
        let mut dropout = 0.0;
        let mut states = ptr::null_mut();
        let mut seed = 0;
        let mut use_mask = false;
        let mut state_evo = false;
        let mut rng_mode = 0;

        let status = unsafe {
            ffi::miopenGetDropoutDescriptor(
                self.desc,
                handle.as_raw(),
                &mut dropout,
                &mut states,
                &mut seed,
                &mut use_mask,
                &mut state_evo,
                &mut rng_mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((dropout, states, seed, use_mask, state_evo, rng_mode))
    }

    /// Get the reserve space size required for dropout operations
    pub fn get_reserve_space_size(x_desc: &TensorDescriptor) -> Result<usize> {
        let mut reserve_space_size_in_bytes = 0;

        let status = unsafe {
            ffi::miopenDropoutGetReserveSpaceSize(x_desc.as_raw(), &mut reserve_space_size_in_bytes)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(reserve_space_size_in_bytes)
    }

    /// Get the states size required for dropout operations
    pub fn get_states_size(handle: &Handle) -> Result<usize> {
        let mut state_size_in_bytes = 0;

        let status =
            unsafe { ffi::miopenDropoutGetStatesSize(handle.as_raw(), &mut state_size_in_bytes) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(state_size_in_bytes)
    }

    /// Execute a forward dropout operation
    pub unsafe fn forward(
        &self,
        handle: &Handle,
        noise_shape: &TensorDescriptor,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        y_desc: &TensorDescriptor,
        y: *mut c_void,
        reserve_space: *mut c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenDropoutForward(
                handle.as_raw(),
                self.desc,
                noise_shape.as_raw(),
                x_desc.as_raw(),
                x,
                y_desc.as_raw(),
                y,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute a backward dropout operation
    pub unsafe fn backward(
        &self,
        handle: &Handle,
        noise_shape: &TensorDescriptor,
        dy_desc: &TensorDescriptor,
        dy: *const c_void,
        dx_desc: &TensorDescriptor,
        dx: *mut c_void,
        reserve_space: *mut c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenDropoutBackward(
                handle.as_raw(),
                self.desc,
                noise_shape.as_raw(),
                dy_desc.as_raw(),
                dy,
                dx_desc.as_raw(),
                dx,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenDropoutDescriptor_t {
        self.desc
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyDropoutDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}
