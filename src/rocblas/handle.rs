/*!
# rocBLAS Context Handle

This module provides a safe wrapper for the rocBLAS handle which is required
for all library operations.
*/

use std::ptr;
use std::marker::PhantomData;
use crate::rocblas::error::{Error, Result, check_error};
use crate::rocblas::{bindings, ffi, hipStream_t};
use crate::rocblas::types::{PointerMode, AtomicsMode, PerformanceMetric, MathMode};

/// Safe wrapper for the rocBLAS handle
pub struct Handle {
    handle: ffi::rocblas_handle,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl Handle {
    /// Create a new rocBLAS handle
    ///
    /// # Returns
    ///
    /// A result containing the newly created handle or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// ```
    pub fn new() -> Result<Self> {
        let mut handle: ffi::rocblas_handle = ptr::null_mut();

        unsafe {
            check_error(ffi::rocblas_create_handle(&mut handle))?;
        }

        Ok(Handle {
            handle,
            _marker: PhantomData,
        })
    }

    /// Set the ROCm/HIP stream for this handle
    ///
    /// # Arguments
    ///
    /// * `stream` - ROCm/HIP stream to use
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let mut handle = Handle::new()?;
    /// let stream = /* get HIP stream */;
    /// handle.set_stream(stream)?;
    /// ```
    pub fn set_stream(&mut self, stream: hipStream_t) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocblas_set_stream(self.handle, stream))
        }
    }

    /// Get the current ROCm/HIP stream used by this handle
    ///
    /// # Returns
    ///
    /// A result containing the current stream or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// let stream = handle.get_stream()?;
    /// ```
    pub fn get_stream(&self) -> Result<hipStream_t> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let mut stream: hipStream_t = ptr::null_mut();

        unsafe {
            check_error(ffi::rocblas_get_stream(self.handle, &mut stream as *mut bindings::hipStream_t))?;
        }

        Ok(stream)
    }

    /// Set the pointer mode for this handle
    ///
    /// # Arguments
    ///
    /// * `mode` - Pointer mode (host or device)
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::{Handle, PointerMode};
    ///
    /// let mut handle = Handle::new()?;
    /// handle.set_pointer_mode(PointerMode::Host)?;
    /// ```
    pub fn set_pointer_mode(&mut self, mode: PointerMode) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocblas_set_pointer_mode(self.handle, mode.into()))
        }
    }

    /// Get the current pointer mode for this handle
    ///
    /// # Returns
    ///
    /// A result containing the current pointer mode or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// let mode = handle.get_pointer_mode()?;
    /// ```
    pub fn get_pointer_mode(&self) -> Result<PointerMode> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let mut mode: u32 = 0;

        unsafe {
            check_error(ffi::rocblas_get_pointer_mode(self.handle, &mut mode))?;
        }

        Ok(mode.into())
    }

    /// Set the atomics mode for this handle
    ///
    /// # Arguments
    ///
    /// * `mode` - Atomics mode (allowed or not allowed)
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::{Handle, AtomicsMode};
    ///
    /// let mut handle = Handle::new()?;
    /// handle.set_atomics_mode(AtomicsMode::Allowed)?;
    /// ```
    pub fn set_atomics_mode(&mut self, mode: AtomicsMode) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocblas_set_atomics_mode(self.handle, mode.into()))
        }
    }

    /// Get the current atomics mode for this handle
    ///
    /// # Returns
    ///
    /// A result containing the current atomics mode or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// let mode = handle.get_atomics_mode()?;
    /// ```
    pub fn get_atomics_mode(&self) -> Result<AtomicsMode> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let mut mode: u32 = 0;

        unsafe {
            check_error(ffi::rocblas_get_atomics_mode(self.handle, &mut mode))?;
        }

        Ok(mode.into())
    }

    /// Set the math mode for this handle
    ///
    /// # Arguments
    ///
    /// * `mode` - Math mode to use
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::{Handle, MathMode};
    ///
    /// let mut handle = Handle::new()?;
    /// handle.set_math_mode(MathMode::DefaultMath)?;
    /// ```
    pub fn set_math_mode(&mut self, mode: MathMode) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocblas_set_math_mode(self.handle, mode.into()))
        }
    }

    /// Get the current math mode for this handle
    ///
    /// # Returns
    ///
    /// A result containing the current math mode or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// let mode = handle.get_math_mode()?;
    /// ```
    pub fn get_math_mode(&self) -> Result<MathMode> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let mut mode: u32 = 0;

        unsafe {
            check_error(ffi::rocblas_get_math_mode(self.handle, &mut mode))?;
        }

        Ok(mode.into())
    }

    /// Set the performance metric for this handle
    ///
    /// # Arguments
    ///
    /// * `metric` - Performance metric to use for solution selection
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::{Handle, PerformanceMetric};
    ///
    /// let mut handle = Handle::new()?;
    /// handle.set_performance_metric(PerformanceMetric::DeviceEfficiency)?;
    /// ```
    pub fn set_performance_metric(&mut self, metric: PerformanceMetric) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocblas_set_performance_metric(self.handle, metric.into()))
        }
    }

    /// Get the current performance metric for this handle
    ///
    /// # Returns
    ///
    /// A result containing the current performance metric or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rocm_rs::rocblas::Handle;
    ///
    /// let handle = Handle::new()?;
    /// let metric = handle.get_performance_metric()?;
    /// ```
    pub fn get_performance_metric(&self) -> Result<PerformanceMetric> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        let mut metric: u32 = 0;

        unsafe {
            check_error(ffi::rocblas_get_performance_metric(self.handle, &mut metric))?;
        }

        Ok(metric.into())
    }

    /// Get the internal handle (for use in other rocBLAS functions)
    pub(crate) fn as_ptr(&self) -> ffi::rocblas_handle {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::rocblas_destroy_handle(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

// Prevent sending a handle between threads as it's not guaranteed to be thread-safe