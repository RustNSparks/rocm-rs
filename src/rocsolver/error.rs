// src/rocsolver/error.rs

use std::fmt;
use std::error::Error as StdError;
use crate::rocblas::ffi::rocblas_status;

/// Error type for RocSOLVER operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: rocblas_status,
}

/// Result type for RocSOLVER operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a RocBLAS error code
    pub fn new(code: rocblas_status) -> Self {
        Self { code }
    }

    /// Convert a RocBLAS error code to a Result
    pub fn from_rocblas_error<T>(error: rocblas_status) -> Result<T>
    where
        T: Default,
    {
        if error == crate::rocblas::ffi::rocblas_status__rocblas_status_success {
            Ok(T::default())
        } else {
            Err(Error::new(error))
        }
    }

    /// Convert a RocBLAS error code to a Result with a specific value
    pub fn from_rocblas_error_with_value<T>(error: rocblas_status, value: T) -> Result<T> {
        if error == crate::rocblas::ffi::rocblas_status__rocblas_status_success {
            Ok(value)
        } else {
            Err(Error::new(error))
        }
    }

    /// Get the raw error code
    pub fn code(&self) -> rocblas_status {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RocSOLVER error code: {}", self.code)
    }
}

impl StdError for Error {}