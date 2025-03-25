/*!
# Kernel cache management for rocFFT

This module provides functions to serialize and deserialize the rocFFT
compiled kernel cache, allowing kernel caches to be saved and loaded
between application runs.
*/

use std::ptr;
use std::slice;
use crate::rocfft::error::{Error, Result, check_error};
use crate::rocfft::bindings;

/// A buffer containing serialized kernel cache data
pub struct CacheBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

impl CacheBuffer {
    /// Get a slice of the buffer contents
    pub fn as_slice(&self) -> &[u8] {
        if self.ptr.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe {
                slice::from_raw_parts(self.ptr as *const u8, self.len)
            }
        }
    }

    /// Get the length of the buffer in bytes
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.ptr.is_null()
    }
}

impl Drop for CacheBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                bindings::rocfft_cache_buffer_free(self.ptr);
            }
            self.ptr = ptr::null_mut();
            self.len = 0;
        }
    }
}

/// Serialize the current compiled kernel cache into a buffer
///
/// This function captures the current state of the rocFFT compiled kernel cache
/// and serializes it into a buffer that can be saved and later deserialized.
/// This can significantly improve startup performance for applications that
/// use the same FFT configurations repeatedly.
///
/// # Returns
///
/// A result containing the serialized cache buffer
///
/// # Example
///
/// ```no_run
/// use crate::rocfft;
/// use std::fs::File;
/// use std::io::Write;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     rocfft::setup()?;
///
///     // After running some transforms, serialize the cache
///     let buffer = rocfft::cache::serialize()?;
///
///     // Save to a file
///     let mut file = File::create("rocfft_cache.bin")?;
///     file.write_all(buffer.as_slice())?;
///
///     rocfft::cleanup()?;
///     Ok(())
/// }
/// ```
pub fn serialize() -> Result<CacheBuffer> {
    let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
    let mut len: usize = 0;

    unsafe {
        check_error(bindings::rocfft_cache_serialize(&mut ptr, &mut len))?;
    }

    Ok(CacheBuffer { ptr, len })
}

/// Deserialize a buffer into the compiled kernel cache
///
/// This function loads a previously serialized kernel cache into the rocFFT
/// runtime, which can avoid recompilation of kernels and improve startup
/// performance.
///
/// # Arguments
///
/// * `data` - Slice containing the serialized cache data
///
/// # Returns
///
/// A result indicating success or an error
///
/// # Example
///
/// ```no_run
/// use crate::rocfft;
/// use std::fs::File;
/// use std::io::Read;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Load from a file before initializing rocFFT
///     let mut file = File::open("rocfft_cache.bin")?;
///     let mut data = Vec::new();
///     file.read_to_end(&mut data)?;
///
///     rocfft::setup()?;
///
///     // Deserialize into the cache
///     rocfft::cache::deserialize(&data)?;
///     
///     // Now use rocFFT with precompiled kernels
///     
///     rocfft::cleanup()?;
///     Ok(())
/// }
/// ```
pub fn deserialize(data: &[u8]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    unsafe {
        check_error(bindings::rocfft_cache_deserialize(
            data.as_ptr() as *const std::ffi::c_void,
            data.len(),
        ))
    }
}