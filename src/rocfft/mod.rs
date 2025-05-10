// src/rocfft/mod.rs

//! Bindings for rocfft
//! Auto-generated - do not modify
#[allow(warnings)]
pub mod bindings;
pub mod cache;
pub mod description;
pub mod error;
pub mod execution;
pub mod ffi;
pub mod field;
pub mod plan;

// Add the new utility modules
pub mod examples;
pub mod utils;

// Re-export all bindings
pub use bindings::*;

/// Initialize rocFFT library
pub fn setup() -> error::Result<()> {
    unsafe { error::check_error(bindings::rocfft_setup()) }
}

/// Cleanup rocFFT library
pub fn cleanup() -> error::Result<()> {
    unsafe { error::check_error(bindings::rocfft_cleanup()) }
}

/// Get the rocFFT version string
pub fn get_version() -> error::Result<String> {
    let mut buffer = vec![0u8; 100];
    unsafe {
        error::check_error(bindings::rocfft_get_version_string(
            buffer.as_mut_ptr() as *mut i8,
            buffer.len(),
        ))?;

        // Find the null terminator
        let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        buffer.truncate(len);

        Ok(String::from_utf8_lossy(&buffer).to_string())
    }
}
