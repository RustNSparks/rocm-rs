// src/hip/error.rs

use crate::hip::ffi;
use std::error::Error as StdError;
use std::fmt;

/// Error type for HIP operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: ffi::hipError_t,
}

/// Result type for HIP operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a HIP error code
    pub fn new(code: ffi::hipError_t) -> Self {
        Self { code }
    }

    /// Convert a HIP error code to a Result
    pub fn from_hip_error<T>(error: ffi::hipError_t) -> Result<T>
    where
        T: Default,
    {
        if error == ffi::hipError_t_hipSuccess {
            Ok(T::default())
        } else {
            Err(Error::new(error))
        }
    }

    /// Convert a HIP error code to a Result with a specific value
    pub fn from_hip_error_with_value<T>(error: ffi::hipError_t, value: T) -> Result<T> {
        if error == ffi::hipError_t_hipSuccess {
            Ok(value)
        } else {
            Err(Error::new(error))
        }
    }

    /// Returns true if the error code represents success
    pub fn is_success(&self) -> bool {
        self.code == ffi::hipError_t_hipSuccess
    }

    /// Get the raw error code
    pub fn code(&self) -> ffi::hipError_t {
        self.code
    }

    /// Returns the error name as a string
    pub fn name(&self) -> &'static str {
        unsafe {
            let name_ptr = ffi::hipGetErrorName(self.code);
            if name_ptr.is_null() {
                "Unknown error"
            } else {
                // This is safe because hipGetErrorName returns a static string
                std::ffi::CStr::from_ptr(name_ptr)
                    .to_str()
                    .unwrap_or("Invalid error string")
            }
        }
    }

    /// Returns the error description as a string
    pub fn description(&self) -> &'static str {
        unsafe {
            let desc_ptr = ffi::hipGetErrorString(self.code);
            if desc_ptr.is_null() {
                "Unknown error"
            } else {
                // This is safe because hipGetErrorString returns a static string
                std::ffi::CStr::from_ptr(desc_ptr)
                    .to_str()
                    .unwrap_or("Invalid error string")
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HIP error {}: {} - {}",
            self.code,
            self.name(),
            self.description()
        )
    }
}

impl StdError for Error {}

// Define error conversion functions for common HIP error codes
impl Error {
    pub fn is_invalid_value(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorInvalidValue
    }

    pub fn is_out_of_memory(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorOutOfMemory
            || self.code == ffi::hipError_t_hipErrorMemoryAllocation
    }

    pub fn is_not_initialized(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorNotInitialized
    }

    pub fn is_invalid_device(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorInvalidDevice
    }

    pub fn is_invalid_context(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorInvalidContext
    }

    pub fn is_not_ready(&self) -> bool {
        self.code == ffi::hipError_t_hipErrorNotReady
    }
}
