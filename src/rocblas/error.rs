// src/rocblas/error.rs

use std::fmt;
use std::error::Error as StdError;
use crate::rocblas::ffi;

/// Error type for RocBLAS operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: ffi::rocblas_status,
}

/// Result type for RocBLAS operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a RocBLAS error code
    pub fn new(code: ffi::rocblas_status) -> Self {
        Self { code }
    }

    /// Convert a RocBLAS error code to a Result
    pub fn from_rocblas_error<T>(error: ffi::rocblas_status) -> Result<T>
    where
        T: Default,
    {
        if error == ffi::rocblas_status__rocblas_status_success {
            Ok(T::default())
        } else {
            Err(Error::new(error))
        }
    }

    /// Convert a RocBLAS error code to a Result with a specific value
    pub fn from_rocblas_error_with_value<T>(error: ffi::rocblas_status, value: T) -> Result<T> {
        if error == ffi::rocblas_status__rocblas_status_success {
            Ok(value)
        } else {
            Err(Error::new(error))
        }
    }

    /// Returns true if the error code represents success
    pub fn is_success(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_success
    }

    /// Get the raw error code
    pub fn code(&self) -> ffi::rocblas_status {
        self.code
    }

    /// Get the name of the error code
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
            ffi::rocblas_status__rocblas_status_size_query_mismatch => "rocblas_status_size_query_mismatch",
            ffi::rocblas_status__rocblas_status_size_increased => "rocblas_status_size_increased",
            ffi::rocblas_status__rocblas_status_size_unchanged => "rocblas_status_size_unchanged",
            ffi::rocblas_status__rocblas_status_invalid_value => "rocblas_status_invalid_value",
            ffi::rocblas_status__rocblas_status_continue => "rocblas_status_continue",
            ffi::rocblas_status__rocblas_status_check_numerics_fail => "rocblas_status_check_numerics_fail",
            ffi::rocblas_status__rocblas_status_excluded_from_build => "rocblas_status_excluded_from_build",
            ffi::rocblas_status__rocblas_status_arch_mismatch => "rocblas_status_arch_mismatch",
            _ => "Unknown rocblas_status code",
        }
    }

    /// Get the description of the error code
    pub fn description(&self) -> &'static str {
        match self.code {
            ffi::rocblas_status__rocblas_status_success => "Success",
            ffi::rocblas_status__rocblas_status_invalid_handle => "Handle not initialized, invalid, or null",
            ffi::rocblas_status__rocblas_status_not_implemented => "Function is not implemented",
            ffi::rocblas_status__rocblas_status_invalid_pointer => "Invalid pointer argument",
            ffi::rocblas_status__rocblas_status_invalid_size => "Invalid size argument",
            ffi::rocblas_status__rocblas_status_memory_error => "Failed internal memory allocation, copy, or dealloc",
            ffi::rocblas_status__rocblas_status_internal_error => "Other internal library failure",
            ffi::rocblas_status__rocblas_status_perf_degraded => "Performance degraded due to low device memory",
            ffi::rocblas_status__rocblas_status_size_query_mismatch => "Unmatched start/stop size query",
            ffi::rocblas_status__rocblas_status_size_increased => "Queried device memory size increased",
            ffi::rocblas_status__rocblas_status_size_unchanged => "Queried device memory size unchanged",
            ffi::rocblas_status__rocblas_status_invalid_value => "Passed argument not valid",
            ffi::rocblas_status__rocblas_status_continue => "Nothing preventing function to proceed",
            ffi::rocblas_status__rocblas_status_check_numerics_fail => "Check numerics failure",
            ffi::rocblas_status__rocblas_status_excluded_from_build => "Feature excluded from build",
            ffi::rocblas_status__rocblas_status_arch_mismatch => "Architecture mismatch",
            _ => "Unknown error",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RocBLAS error {}: {} - {}", self.code, self.name(), self.description())
    }
}

impl StdError for Error {}

// Define error conversion functions for common RocBLAS error codes
impl Error {
    pub fn is_invalid_handle(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_handle
    }

    pub fn is_not_implemented(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_not_implemented
    }

    pub fn is_invalid_pointer(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_pointer
    }

    pub fn is_invalid_size(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_size
    }

    pub fn is_memory_error(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_memory_error
    }

    pub fn is_internal_error(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_internal_error
    }

    pub fn is_invalid_value(&self) -> bool {
        self.code == ffi::rocblas_status__rocblas_status_invalid_value
    }
}