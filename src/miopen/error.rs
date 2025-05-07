// src/miopen/error.rs

use crate::miopen::ffi;
use std::error::Error as StdError;
use std::fmt;

/// Error type for MIOpen operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: ffi::miopenStatus_t,
}

/// Result type for MIOpen operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a MIOpen status code
    pub fn new(code: ffi::miopenStatus_t) -> Self {
        Self { code }
    }

    /// Convert a MIOpen status code to a Result
    pub fn from_miopen_status<T>(status: ffi::miopenStatus_t) -> Result<T>
    where
        T: Default,
    {
        if status == ffi::miopenStatus_t_miopenStatusSuccess {
            Ok(T::default())
        } else {
            Err(Error::new(status))
        }
    }

    /// Convert a MIOpen status code to a Result with a specific value
    pub fn from_miopen_status_with_value<T>(status: ffi::miopenStatus_t, value: T) -> Result<T> {
        if status == ffi::miopenStatus_t_miopenStatusSuccess {
            Ok(value)
        } else {
            Err(Error::new(status))
        }
    }

    /// Returns true if the status code represents success
    pub fn is_success(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusSuccess
    }

    /// Get the raw status code
    pub fn code(&self) -> ffi::miopenStatus_t {
        self.code
    }

    /// Returns the error description as a string
    pub fn description(&self) -> &'static str {
        unsafe {
            let desc_ptr = ffi::miopenGetErrorString(self.code);
            if desc_ptr.is_null() {
                "Unknown error"
            } else {
                // This is safe because miopenGetErrorString returns a static string
                std::ffi::CStr::from_ptr(desc_ptr)
                    .to_str()
                    .unwrap_or("Invalid error string")
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MIOpen error {}: {}", self.code, self.description())
    }
}

impl StdError for Error {}

// Define error conversion functions for common MIOpen error codes
impl Error {
    pub fn is_not_initialized(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusNotInitialized
    }

    pub fn is_invalid_value(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusInvalidValue
    }

    pub fn is_bad_param(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusBadParm
    }

    pub fn is_alloc_failed(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusAllocFailed
    }

    pub fn is_internal_error(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusInternalError
    }

    pub fn is_not_implemented(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusNotImplemented
    }

    pub fn is_unsupported_op(&self) -> bool {
        self.code == ffi::miopenStatus_t_miopenStatusUnsupportedOp
    }
}
