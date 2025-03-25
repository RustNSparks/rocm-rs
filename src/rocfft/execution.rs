use std::ptr;
use std::marker::PhantomData;
use crate::rocfft::error::{Error, Result, check_error};
use crate::rocfft::ffi;

/// Additional execution parameters for a transform
///
/// This structure can control:
/// - Work buffers
/// - Execution streams (HIP/ROCm streams)
/// - Load/store callbacks
pub struct ExecutionInfo {
    handle: ffi::rocfft_execution_info,
    _marker: PhantomData<*mut ()>, // Mark as !Send and !Sync
}

impl ExecutionInfo {
    /// Create a new execution info object
    ///
    /// # Returns
    ///
    /// A result containing the newly created execution info or an error
    pub fn new() -> Result<Self> {
        let mut handle: ffi::rocfft_execution_info = ptr::null_mut();

        unsafe {
            check_error(ffi::rocfft_execution_info_create(&mut handle))?;
        }

        Ok(ExecutionInfo {
            handle,
            _marker: PhantomData,
        })
    }

    /// Set a work buffer for the transform
    ///
    /// # Arguments
    ///
    /// * `buffer` - Pointer to work buffer (GPU memory)
    /// * `size_in_bytes` - Size of work buffer in bytes
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Note
    ///
    /// If you need to know how large the work buffer should be, call 
    /// `Plan::get_work_buffer_size()`.
    pub fn set_work_buffer(
        &mut self,
        buffer: *mut std::ffi::c_void,
        size_in_bytes: usize,
    ) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        if buffer.is_null() && size_in_bytes > 0 {
            return Err(Error::NullPointer);
        }

        unsafe {
            check_error(ffi::rocfft_execution_info_set_work_buffer(
                self.handle,
                buffer,
                size_in_bytes,
            ))
        }
    }

    /// Set a ROCm/HIP stream for the transform execution
    ///
    /// # Arguments
    ///
    /// * `stream` - HIP stream to use (hipStream_t)
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    pub fn set_stream(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocfft_execution_info_set_stream(
                self.handle,
                stream,
            ))
        }
    }

    /// Set a load callback for the transform (experimental)
    ///
    /// # Arguments
    ///
    /// * `callbacks` - Array of callback function pointers
    /// * `user_data` - Array of user data pointers passed to callbacks
    /// * `shared_mem_bytes` - Amount of shared memory for the callback
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Note
    ///
    /// This is an experimental feature in rocFFT.
    pub fn set_load_callback(
        &mut self,
        callbacks: &mut [*mut std::ffi::c_void],
        user_data: &mut [*mut std::ffi::c_void],
        shared_mem_bytes: usize,
    ) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocfft_execution_info_set_load_callback(
                self.handle,
                callbacks.as_mut_ptr(),
                user_data.as_mut_ptr(),
                shared_mem_bytes,
            ))
        }
    }

    /// Set a store callback for the transform (experimental)
    ///
    /// # Arguments
    ///
    /// * `callbacks` - Array of callback function pointers
    /// * `user_data` - Array of user data pointers passed to callbacks
    /// * `shared_mem_bytes` - Amount of shared memory for the callback
    ///
    /// # Returns
    ///
    /// A result indicating success or an error
    ///
    /// # Note
    ///
    /// This is an experimental feature in rocFFT.
    pub fn set_store_callback(
        &mut self,
        callbacks: &mut [*mut std::ffi::c_void],
        user_data: &mut [*mut std::ffi::c_void],
        shared_mem_bytes: usize,
    ) -> Result<()> {
        if self.handle.is_null() {
            return Err(Error::ObjectDestroyed);
        }

        unsafe {
            check_error(ffi::rocfft_execution_info_set_store_callback(
                self.handle,
                callbacks.as_mut_ptr(),
                user_data.as_mut_ptr(),
                shared_mem_bytes,
            ))
        }
    }

    /// Get the internal handle (for use in other rocFFT functions)
    pub(crate) fn as_ptr(&self) -> ffi::rocfft_execution_info {
        self.handle
    }
}

impl Drop for ExecutionInfo {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::rocfft_execution_info_destroy(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

// Prevent sending an execution info between threads as it's not guaranteed to be thread-safe
