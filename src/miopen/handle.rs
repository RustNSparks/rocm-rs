// src/miopen/handle.rs

use crate::hip::{Stream, bindings::hipStream_t};
use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use std::ptr;

/// Safe wrapper for MIOpen handle
pub struct Handle {
    handle: ffi::miopenHandle_t,
}

impl Handle {
    /// Create a new MIOpen handle
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        let status = unsafe { ffi::miopenCreate(&mut handle) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { handle })
    }

    /// Create a new MIOpen handle with a stream
    pub fn with_stream(stream: &Stream) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let status = unsafe {
            ffi::miopenCreateWithStream(
                &mut handle,
                stream.as_raw() as crate::miopen::bindings::miopenAcceleratorQueue_t,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { handle })
    }

    /// Set the stream for this handle
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetStream(
                self.handle,
                stream.as_raw() as crate::miopen::bindings::miopenAcceleratorQueue_t,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the current stream for this handle
    pub fn get_stream(&self) -> Result<Stream> {
        let mut stream_id = ptr::null_mut();
        let status = unsafe { ffi::miopenGetStream(self.handle, &mut stream_id) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        // Create a stream from the raw pointer
        Ok(Stream::from_raw(stream_id as hipStream_t))
    }

    /// Enable or disable profiling
    pub fn enable_profiling(&self, enable: bool) -> Result<()> {
        let status = unsafe { ffi::miopenEnableProfiling(self.handle, enable) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the timing of the last kernel executed
    pub fn get_kernel_time(&self) -> Result<f32> {
        let mut time = 0.0;
        let status = unsafe { ffi::miopenGetKernelTime(self.handle, &mut time) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(time)
    }

    /// Set a custom allocator for MIOpen
    pub unsafe fn set_allocator(
        &self,
        allocator: ffi::miopenAllocatorFunction,
        deallocator: ffi::miopenDeallocatorFunction,
        context: *mut ::std::os::raw::c_void,
    ) -> Result<()> {
        let status =
            unsafe { ffi::miopenSetAllocator(self.handle, allocator, deallocator, context) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw handle
    pub fn as_raw(&self) -> ffi::miopenHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::miopenDestroy(self.handle);
            };
            self.handle = ptr::null_mut();
        }
    }
}
