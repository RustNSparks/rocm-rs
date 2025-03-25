//! Bindings for rocfft
//! Auto-generated - do not modify

pub mod bindings;
pub mod field;
pub mod plan;
pub mod description;
pub mod error;
pub mod ffi;
pub mod cache;
pub mod execution;

// Re-export commonly used types
pub use bindings::*;

use error::*;

/// Initialize the rocFFT library
///
/// This function must be called before any other rocFFT functions.
///
/// # Returns
///
/// A `Result` indicating success or the specific error that occurred.
///
/// # Example
///
/// ```no_run
/// use crate::rocfft;
///
/// fn main() -> rocfft::Result<()> {
///     rocfft::setup()?;
///     // Use rocFFT here...
///     rocfft::cleanup()?;
///     Ok(())
/// }
/// ```
pub fn setup() -> Result<()> {
    unsafe {
        error::check_error(bindings::rocfft_setup())
    }
}

/// Clean up the rocFFT library
///
/// This function should be called after all rocFFT operations are complete.
///
/// # Returns
///
/// A `Result` indicating success or the specific error that occurred.
///
/// # Example
///
/// ```no_run
/// use crate::rocfft;
///
/// fn main() -> rocfft::Result<()> {
///     rocfft::setup()?;
///     // Use rocFFT here...
///     rocfft::cleanup()?;
///     Ok(())
/// }
/// ```
pub fn cleanup() -> Result<()> {
    unsafe {
        error::check_error(bindings::rocfft_cleanup())
    }
}

/// Get the rocFFT library version string
///
/// # Returns
///
/// A `Result` containing the version string.
///
/// # Example
///
/// ```no_run
///
/// fn main() -> rocfft::Result<()> {
///     use rocm_rs::rocfft;
/// rocfft::setup()?;
///     let version = rocfft::get_version()?;
///     println!("rocFFT version: {}", version);
///     rocfft::cleanup()?;
///     Ok(())
/// }
/// ```
pub fn get_version() -> Result<String> {
    let mut buffer = vec![0u8; 128];
    unsafe {
        error::check_error(bindings::rocfft_get_version_string(
            buffer.as_mut_ptr() as *mut std::os::raw::c_char,
            buffer.len()
        ))?;
    }

    // Find the null terminator
    let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());

    // Convert to a valid UTF-8 string
    let version = String::from_utf8_lossy(&buffer[0..end]).to_string();
    Ok(version)
}