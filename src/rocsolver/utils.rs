// src/rocsolver/utils.rs

use std::ffi::{c_char, CStr, CString};
use crate::rocblas;
use crate::rocblas::ffi;
use crate::rocsolver::error::{Error, Result};

/// Gets the RocSOLVER library version string
pub fn get_version_string() -> Result<String> {
    unsafe {
        // First get the required buffer size
        let mut size = 0;
        let status = ffi::rocsolver_get_version_string_size(&mut size);
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }

        // Allocate buffer and get the string
        let mut buffer = vec![0u8; size];
        let status = ffi::rocsolver_get_version_string(
            buffer.as_mut_ptr() as *mut c_char, 
            size
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }

        // Convert to Rust string
        let c_str = CStr::from_bytes_with_nul(&buffer)
            .map_err(|_| Error::new(ffi::rocblas_status__rocblas_status_internal_error))?;
        
        Ok(c_str.to_string_lossy().into_owned())
    }
}

/// Begins a RocSOLVER multi-level logging session
pub fn log_begin() -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_begin() };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Ends the multi-level RocSOLVER logging session
pub fn log_end() -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_end() };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Sets the logging mode for the RocSOLVER multi-level logging environment
pub fn log_set_layer_mode(layer_mode: u32) -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_set_layer_mode(layer_mode) };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Sets the maximum trace log depth for the RocSOLVER multi-level logging environment
pub fn log_set_max_levels(max_levels: i32) -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_set_max_levels(max_levels) };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Restores the default values of the RocSOLVER multi-level logging environment
pub fn log_restore_defaults() -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_restore_defaults() };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Prints the profile logging results
pub fn log_write_profile() -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_write_profile() };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Prints the profile logging results and clears the profile record
pub fn log_flush_profile() -> Result<()> {
    let status = unsafe { ffi::rocsolver_log_flush_profile() };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}