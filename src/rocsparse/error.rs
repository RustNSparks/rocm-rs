//! Error types for ROCsparse operations

use crate::rocsparse::rocsparse_status;

use super::bindings;

/// Error type for ROCsparse operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidHandle,
    NotImplemented,
    InvalidPointer,
    InvalidSize,
    MemoryError,
    InternalError,
    InvalidValue,
    ArchMismatch,
    ZeroPivot,
    NotInitialized,
    TypeMismatch,
    RequiresSortedStorage,
    ThrownException,
    Continue, // This is not an error but part of the status enum
    Unknown(i32),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidHandle => write!(f, "ROCsparse: handle not initialized, invalid or null"),
            Error::NotImplemented => write!(f, "ROCsparse: function is not implemented"),
            Error::InvalidPointer => write!(f, "ROCsparse: invalid pointer parameter"),
            Error::InvalidSize => write!(f, "ROCsparse: invalid size parameter"),
            Error::MemoryError => write!(f, "ROCsparse: failed memory allocation, copy, dealloc"),
            Error::InternalError => write!(f, "ROCsparse: other internal library failure"),
            Error::InvalidValue => write!(f, "ROCsparse: invalid value parameter"),
            Error::ArchMismatch => write!(f, "ROCsparse: device arch is not supported"),
            Error::ZeroPivot => write!(f, "ROCsparse: encountered zero pivot"),
            Error::NotInitialized => write!(f, "ROCsparse: descriptor has not been initialized"),
            Error::TypeMismatch => write!(f, "ROCsparse: index types do not match"),
            Error::RequiresSortedStorage => write!(f, "ROCsparse: sorted storage required"),
            Error::ThrownException => write!(f, "ROCsparse: exception being thrown"),
            Error::Continue => write!(f, "ROCsparse: nothing preventing function to proceed"),
            Error::Unknown(code) => write!(f, "ROCsparse: unknown error code: {}", code),
        }
    }
}

impl std::error::Error for Error {}

/// Alias for Result with ROCsparse error
pub type Result<T> = std::result::Result<T, Error>;

/// Convert low-level status to Result
pub(crate) fn status_to_result(status: rocsparse_status) -> Result<()> {
    match status {
        bindings::rocsparse_status__rocsparse_status_success => Ok(()),
        bindings::rocsparse_status__rocsparse_status_invalid_handle => Err(Error::InvalidHandle),
        bindings::rocsparse_status__rocsparse_status_not_implemented => Err(Error::NotImplemented),
        bindings::rocsparse_status__rocsparse_status_invalid_pointer => Err(Error::InvalidPointer),
        bindings::rocsparse_status__rocsparse_status_invalid_size => Err(Error::InvalidSize),
        bindings::rocsparse_status__rocsparse_status_memory_error => Err(Error::MemoryError),
        bindings::rocsparse_status__rocsparse_status_internal_error => Err(Error::InternalError),
        bindings::rocsparse_status__rocsparse_status_invalid_value => Err(Error::InvalidValue),
        bindings::rocsparse_status__rocsparse_status_arch_mismatch => Err(Error::ArchMismatch),
        bindings::rocsparse_status__rocsparse_status_zero_pivot => Err(Error::ZeroPivot),
        bindings::rocsparse_status__rocsparse_status_not_initialized => Err(Error::NotInitialized),
        bindings::rocsparse_status__rocsparse_status_type_mismatch => Err(Error::TypeMismatch),
        bindings::rocsparse_status__rocsparse_status_requires_sorted_storage => {
            Err(Error::RequiresSortedStorage)
        }
        bindings::rocsparse_status__rocsparse_status_thrown_exception => Err(Error::ThrownException),
        bindings::rocsparse_status__rocsparse_status_continue => Err(Error::Continue),
        _ => Err(Error::Unknown(status as i32)),
    }
}
