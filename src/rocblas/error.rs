use crate::rocblas::bindings;
use std::error::Error as StdError;
use std::ffi::NulError;
use std::fmt;

/// Custom error type for rocBLAS operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Invalid rocBLAS handle
    InvalidHandle,
    /// Function is not implemented
    NotImplemented,
    /// Invalid pointer was passed
    InvalidPointer,
    /// Invalid size parameter
    InvalidSize,
    /// Memory allocation, copy, or deallocation error
    MemoryError,
    /// Internal rocBLAS error
    InternalError,
    /// Performance degraded due to low device memory
    PerfDegraded,
    /// Size query mismatch
    SizeQueryMismatch,
    /// Size increased in query
    SizeIncreased,
    /// Size unchanged in query
    SizeUnchanged,
    /// Invalid value was passed
    InvalidValue,
    /// Check numerics failed
    CheckNumericsFail,
    /// Function excluded from build
    ExcludedFromBuild,
    /// Architecture mismatch
    ArchMismatch,
    /// A null pointer was encountered where a valid pointer was required
    NullPointer,
    /// A non-UTF8 string was encountered
    InvalidString,
    /// An operation was attempted on an object that has already been destroyed
    ObjectDestroyed,
    /// Memory allocation failed
    OutOfMemory,
    /// Error converting to or from a C string
    NulError(String),
    /// Invalid device or device context
    InvalidDevice,
    /// Unsupported combination of parameters
    UnsupportedConfiguration,
    /// Any other unexpected error
    Unknown(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidHandle => write!(f, "Invalid rocBLAS handle"),
            Error::NotImplemented => write!(f, "Function is not implemented"),
            Error::InvalidPointer => write!(f, "Invalid pointer"),
            Error::InvalidSize => write!(f, "Invalid size"),
            Error::MemoryError => write!(f, "Memory error"),
            Error::InternalError => write!(f, "Internal rocBLAS error"),
            Error::PerfDegraded => write!(f, "Performance degraded due to low device memory"),
            Error::SizeQueryMismatch => write!(f, "Size query mismatch"),
            Error::SizeIncreased => write!(f, "Size increased"),
            Error::SizeUnchanged => write!(f, "Size unchanged"),
            Error::InvalidValue => write!(f, "Invalid value"),
            Error::CheckNumericsFail => write!(f, "Check numerics failed"),
            Error::ExcludedFromBuild => write!(f, "Function excluded from build"),
            Error::ArchMismatch => write!(f, "Architecture mismatch"),
            Error::NullPointer => write!(f, "Null pointer"),
            Error::InvalidString => write!(f, "Invalid string"),
            Error::ObjectDestroyed => write!(f, "Object has been destroyed"),
            Error::OutOfMemory => write!(f, "Out of memory"),
            Error::NulError(msg) => write!(f, "C string conversion error: {}", msg),
            Error::InvalidDevice => write!(f, "Invalid device or device context"),
            Error::UnsupportedConfiguration => write!(f, "Unsupported configuration of parameters"),
            Error::Unknown(code) => write!(f, "Unknown rocBLAS error (code: {})", code),
        }
    }
}

impl StdError for Error {}

impl From<NulError> for Error {
    fn from(err: NulError) -> Self {
        Error::NulError(err.to_string())
    }
}

impl From<u32> for Error {
    fn from(status: u32) -> Self {
        match status {
            bindings::rocblas_status__rocblas_status_success => {
                panic!("Tried to convert successful status to error")
            }
            bindings::rocblas_status__rocblas_status_invalid_handle => Error::InvalidHandle,
            bindings::rocblas_status__rocblas_status_not_implemented => Error::NotImplemented,
            bindings::rocblas_status__rocblas_status_invalid_pointer => Error::InvalidPointer,
            bindings::rocblas_status__rocblas_status_invalid_size => Error::InvalidSize,
            bindings::rocblas_status__rocblas_status_memory_error => Error::MemoryError,
            bindings::rocblas_status__rocblas_status_internal_error => Error::InternalError,
            bindings::rocblas_status__rocblas_status_perf_degraded => Error::PerfDegraded,
            bindings::rocblas_status__rocblas_status_size_query_mismatch => {
                Error::SizeQueryMismatch
            }
            bindings::rocblas_status__rocblas_status_size_increased => Error::SizeIncreased,
            bindings::rocblas_status__rocblas_status_size_unchanged => Error::SizeUnchanged,
            bindings::rocblas_status__rocblas_status_invalid_value => Error::InvalidValue,
            bindings::rocblas_status__rocblas_status_check_numerics_fail => {
                Error::CheckNumericsFail
            }
            bindings::rocblas_status__rocblas_status_excluded_from_build => {
                Error::ExcludedFromBuild
            }
            bindings::rocblas_status__rocblas_status_arch_mismatch => Error::ArchMismatch,
            code => Error::Unknown(code),
        }
    }
}

// Convert a string literal to a specific error
impl From<&'static str> for Error {
    fn from(msg: &'static str) -> Self {
        // This function allows convenient creation of errors in the library code
        match msg {
            "null_pointer" => Error::NullPointer,
            "invalid_string" => Error::InvalidString,
            "object_destroyed" => Error::ObjectDestroyed,
            "out_of_memory" => Error::OutOfMemory,
            "invalid_device" => Error::InvalidDevice,
            "unsupported_configuration" => Error::UnsupportedConfiguration,
            _ => Error::InternalError,
        }
    }
}

/// Custom Result type for rocBLAS operations
pub type Result<T> = std::result::Result<T, Error>;

/// Check a rocBLAS status code and convert to a Rust Result
pub(crate) fn check_error(status: u32) -> Result<()> {
    match status {
        bindings::rocblas_status__rocblas_status_success => Ok(()),
        _ => Err(Error::from(status)),
    }
}

/// Validate required pointer arguments and return an error if null
#[inline]
pub(crate) fn check_ptr<T>(ptr: *const T) -> Result<()> {
    if ptr.is_null() {
        Err(Error::NullPointer)
    } else {
        Ok(())
    }
}

/// Validate mutable pointer arguments and return an error if null
#[inline]
pub(crate) fn check_mut_ptr<T>(ptr: *mut T) -> Result<()> {
    if ptr.is_null() {
        Err(Error::NullPointer)
    } else {
        Ok(())
    }
}
