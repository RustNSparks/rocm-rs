// src/rocprofiler/error.rs

use crate::hip;
use std::fmt;
use std::error::Error as StdError;
use std::ffi::CStr;

use super::bindings;

/// Error type for ROCProfiler operations
#[derive(Debug, Clone, Copy)]
pub struct Error {
    status: u32,  // Using hsa_status_t
}

/// Result type for ROCProfiler operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from an HSA status code
    pub fn new(status: u32) -> Self {
        Self { status }
    }

    /// Returns true if the error code represents success
    pub fn is_success(&self) -> bool {
        self.status == bindings::hsa_status_t_HSA_STATUS_SUCCESS
    }

    /// Get the raw error code
    pub fn code(&self) -> u32 {
        self.status
    }

    /// Convert an HSA status code to a Result
    pub fn from_hsa_status<T>(status: u32) -> Result<T>
    where
        T: Default,
    {
        if status == bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            Ok(T::default())
        } else {
            Err(Error::new(status))
        }
    }

    /// Convert an HSA status code to a Result with a specific value
    pub fn from_hsa_status_with_value<T>(status: u32, value: T) -> Result<T> {
        if status == bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            Ok(value)
        } else {
            Err(Error::new(status))
        }
    }

    /// Returns the error description as a string
    pub fn description(&self) -> &'static str {
        match self.status {
            bindings::hsa_status_t_HSA_STATUS_SUCCESS => "Success",
            bindings::hsa_status_t_HSA_STATUS_ERROR => "Generic error",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT => "Invalid argument",
            bindings::hsa_status_t_HSA_STATUS_ERROR_OUT_OF_RESOURCES => "Out of resources",
            bindings::hsa_status_t_HSA_STATUS_ERROR_NOT_INITIALIZED => "Not initialized",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_AGENT => "Invalid agent",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_REGION => "Invalid region",
            _ => unsafe {
                // Try to get the actual error string from ROCProfiler
                let mut error_str_ptr = std::ptr::null();
                if bindings::rocprofiler_error_string(&mut error_str_ptr) == bindings::hsa_status_t_HSA_STATUS_SUCCESS && !error_str_ptr.is_null() {
                    let c_str = CStr::from_ptr(error_str_ptr);
                    match c_str.to_str() {
                        Ok(s) => {
                            // This is not ideal as we're returning a slice that might not live long enough,
                            // but ROCProfiler documentation suggests the string is static
                            s
                        }
                        Err(_) => "Unknown error (invalid UTF-8 in error string)",
                    }
                } else {
                    "Unknown error"
                }
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ROCProfiler error {}: {}", self.status, self.description())
    }
}

impl StdError for Error {}

// Automatic conversion from HIP errors
impl From<hip::Error> for Error {
    fn from(error: hip::Error) -> Self {
        // Map HIP errors to a generic HSA error
        Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR)
    }
}