// src/rocblas/handle.rs

use std::ptr;
use crate::hip::{self, Stream};
use crate::rocblas::ffi;
use crate::rocblas::error::{Error, Result};

/// Safe wrapper for RocBLAS handle
pub struct Handle {
    handle: ffi::rocblas_handle,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    /// Create a new RocBLAS handle
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        let error = unsafe { ffi::rocblas_create_handle(&mut handle) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(Self { handle })
    }

    /// Set the stream for this handle
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        // Use a type cast to convert between the two hipStream_t types
        let hip_stream_ptr = stream.as_raw();
        // Cast to the expected type for rocblas
        let rocblas_stream_ptr = hip_stream_ptr as ffi::hipStream_t;
        
        let error = unsafe { ffi::rocblas_set_stream(self.handle, rocblas_stream_ptr) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(())
    }
    /// Get the stream associated with this handle
    pub fn get_stream(&self) -> Result<Stream> {
        let mut stream_ptr = ptr::null_mut();
        let error = unsafe { ffi::rocblas_get_stream(self.handle, &mut stream_ptr) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        // Cast back to hip::ffi::hipStream_t
        let hip_stream_ptr = stream_ptr as crate::hip::ffi::hipStream_t;
        
        // Create a Stream from the raw pointer
        // This doesn't take ownership of the stream, just wraps the pointer
        Ok(Stream::from_raw(hip_stream_ptr))
    }

    /// Set the pointer mode for this handle
    pub fn set_pointer_mode(&self, mode: ffi::rocblas_pointer_mode) -> Result<()> {
        let error = unsafe { ffi::rocblas_set_pointer_mode(self.handle, mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Get the pointer mode for this handle
    pub fn get_pointer_mode(&self) -> Result<ffi::rocblas_pointer_mode> {
        let mut mode = ffi::rocblas_pointer_mode__rocblas_pointer_mode_host;
        let error = unsafe { ffi::rocblas_get_pointer_mode(self.handle, &mut mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(mode)
    }

    /// Set the atomics mode for this handle
    pub fn set_atomics_mode(&self, mode: ffi::rocblas_atomics_mode) -> Result<()> {
        let error = unsafe { ffi::rocblas_set_atomics_mode(self.handle, mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Get the atomics mode for this handle
    pub fn get_atomics_mode(&self) -> Result<ffi::rocblas_atomics_mode> {
        let mut mode = ffi::rocblas_atomics_mode__rocblas_atomics_allowed;
        let error = unsafe { ffi::rocblas_get_atomics_mode(self.handle, &mut mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(mode)
    }

    /// Set the performance metric for this handle
    pub fn set_performance_metric(&self, metric: ffi::rocblas_performance_metric) -> Result<()> {
        let error = unsafe { ffi::rocblas_set_performance_metric(self.handle, metric) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Get the performance metric for this handle
    pub fn get_performance_metric(&self) -> Result<ffi::rocblas_performance_metric> {
        let mut metric = ffi::rocblas_performance_metric__rocblas_default_performance_metric;
        let error = unsafe { ffi::rocblas_get_performance_metric(self.handle, &mut metric) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(metric)
    }

    /// Set the math mode for this handle
    pub fn set_math_mode(&self, mode: ffi::rocblas_math_mode) -> Result<()> {
        let error = unsafe { ffi::rocblas_set_math_mode(self.handle, mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Get the math mode for this handle
    pub fn get_math_mode(&self) -> Result<ffi::rocblas_math_mode> {
        let mut mode = ffi::rocblas_math_mode__rocblas_default_math;
        let error = unsafe { ffi::rocblas_get_math_mode(self.handle, &mut mode) };

        if error != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(error));
        }

        Ok(mode)
    }

    /// Get the raw handle
    pub fn as_raw(&self) -> ffi::rocblas_handle {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::rocblas_destroy_handle(self.handle);
                // We cannot handle errors in drop, so just ignore the result
            }
            self.handle = ptr::null_mut();
        }
    }
}