// src/rocsolver/error.rs
//! Error handling for rocSOLVER operations.
//!
//! This module provides error types and result aliases for rocSOLVER operations.
//! It wraps the underlying rocblas_status codes with a Rust-friendly API.

use crate::rocblas::ffi;
use std::error::Error as StdError;
use std::fmt;

/// Error type for rocSOLVER operations.
///
/// This type wraps the underlying `rocblas_status` error codes and provides
/// methods for querying error details and converting between error representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: ffi::rocblas_status,
}

/// Result type for rocSOLVER operations.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a rocblas_status code.
    #[inline]
    pub fn new(code: ffi::rocblas_status) -> Self {
        Self { code }
    }

    /// Convert a rocblas_status code to a Result.
    ///
    /// Returns `Ok(T::default())` if the status is success, otherwise returns an error.
    #[inline]
    pub fn from_status<T: Default>(status: ffi::rocblas_status) -> Result<T> {
        if status == ffi::rocblas_status__rocblas_status_success {
            Ok(T::default())
        } else {
            Err(Error::new(status))
        }
    }

    /// Convert a rocblas_status code to a Result with a specific value.
    ///
    /// Returns `Ok(value)` if the status is success, otherwise returns an error.
    #[inline]
    pub fn from_status_with_value<T>(status: ffi::rocblas_status, value: T) -> Result<T> {
        if status == ffi::rocblas_status__rocblas_status_success {
            Ok(value)
        } else {
            Err(Error::new(status))
        }
    }

    /// Returns true if the status code represents success.
    #[inline]
    pub fn is_success(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_success
    }

    /// Get the raw error code.
    #[inline]
    pub fn code(&self) -> ffi::rocblas_status {
        self.code
    }

    /// Get the name of the error code.
    pub fn name(&self) -> &'static str {
        match self.code {
            ffi::rocblas_status__rocblas_status_success => "rocblas_status_success",
            ffi::rocblas_status__rocblas_status_invalid_handle => "rocblas_status_invalid_handle",
            ffi::rocblas_status__rocblas_status_not_implemented => "rocblas_status_not_implemented",
            ffi::rocblas_status__rocblas_status_invalid_pointer => "rocblas_status_invalid_pointer",
            ffi::rocblas_status__rocblas_status_invalid_size => "rocblas_status_invalid_size",
            ffi::rocblas_status__rocblas_status_memory_error => "rocblas_status_memory_error",
            ffi::rocblas_status__rocblas_status_internal_error => "rocblas_status_internal_error",
            ffi::rocblas_status__rocblas_status_perf_degraded => "rocblas_status_perf_degraded",
            ffi::rocblas_status__rocblas_status_size_query_mismatch => {
                "rocblas_status_size_query_mismatch"
            }
            ffi::rocblas_status__rocblas_status_size_increased => "rocblas_status_size_increased",
            ffi::rocblas_status__rocblas_status_size_unchanged => "rocblas_status_size_unchanged",
            ffi::rocblas_status__rocblas_status_invalid_value => "rocblas_status_invalid_value",
            ffi::rocblas_status__rocblas_status_continue => "rocblas_status_continue",
            ffi::rocblas_status__rocblas_status_check_numerics_fail => {
                "rocblas_status_check_numerics_fail"
            }
            ffi::rocblas_status__rocblas_status_excluded_from_build => {
                "rocblas_status_excluded_from_build"
            }
            ffi::rocblas_status__rocblas_status_arch_mismatch => "rocblas_status_arch_mismatch",
            _ => "Unknown status code",
        }
    }

    /// Get a human-readable description of the error.
    pub fn description(&self) -> &'static str {
        match self.code {
            ffi::rocblas_status__rocblas_status_success => "Success",
            ffi::rocblas_status__rocblas_status_invalid_handle => {
                "Handle not initialized, invalid, or null"
            }
            ffi::rocblas_status__rocblas_status_not_implemented => "Function is not implemented",
            ffi::rocblas_status__rocblas_status_invalid_pointer => "Invalid pointer argument",
            ffi::rocblas_status__rocblas_status_invalid_size => "Invalid size argument",
            ffi::rocblas_status__rocblas_status_memory_error => {
                "Failed internal memory allocation, copy, or dealloc"
            }
            ffi::rocblas_status__rocblas_status_internal_error => "Other internal library failure",
            ffi::rocblas_status__rocblas_status_perf_degraded => {
                "Performance degraded due to low device memory"
            }
            ffi::rocblas_status__rocblas_status_size_query_mismatch => {
                "Unmatched start/stop size query"
            }
            ffi::rocblas_status__rocblas_status_size_increased => {
                "Queried device memory size increased"
            }
            ffi::rocblas_status__rocblas_status_size_unchanged => {
                "Queried device memory size unchanged"
            }
            ffi::rocblas_status__rocblas_status_invalid_value => "Passed argument not valid",
            ffi::rocblas_status__rocblas_status_continue => {
                "Nothing preventing function to proceed"
            }
            ffi::rocblas_status__rocblas_status_check_numerics_fail => "Check numerics failure",
            ffi::rocblas_status__rocblas_status_excluded_from_build => {
                "Feature excluded from build"
            }
            ffi::rocblas_status__rocblas_status_arch_mismatch => "Architecture mismatch",
            _ => "Unknown error",
        }
    }

    // Error type checking methods

    /// Returns true if the error is an invalid handle error.
    #[inline]
    pub fn is_invalid_handle(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_handle
    }

    /// Returns true if the error is a not implemented error.
    #[inline]
    pub fn is_not_implemented(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_not_implemented
    }

    /// Returns true if the error is an invalid pointer error.
    #[inline]
    pub fn is_invalid_pointer(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_pointer
    }

    /// Returns true if the error is an invalid size error.
    #[inline]
    pub fn is_invalid_size(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_size
    }

    /// Returns true if the error is a memory error.
    #[inline]
    pub fn is_memory_error(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_memory_error
    }

    /// Returns true if the error is an internal error.
    #[inline]
    pub fn is_internal_error(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_internal_error
    }

    /// Returns true if the error is an invalid value error.
    #[inline]
    pub fn is_invalid_value(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_value
    }

    /// Returns true if the error is a performance degraded warning.
    #[inline]
    pub fn is_perf_degraded(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_perf_degraded
    }

    /// Returns true if the error is an architecture mismatch error.
    #[inline]
    pub fn is_arch_mismatch(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_arch_mismatch
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rocSOLVER error {}: {} - {}",
            self.code,
            self.name(),
            self.description()
        )
    }
}

impl StdError for Error {}

/// Convert from rocblas::Error to rocsolver::Error.
/// Since both use the same underlying status codes, this is a simple conversion.
impl From<crate::rocblas::Error> for Error {
    fn from(err: crate::rocblas::Error) -> Self {
        Error::new(err.code())
    }
}
