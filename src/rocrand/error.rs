// src/rocrand/error.rs

use crate::rocrand::bindings;
use std::fmt;

/// rocRAND error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Header file and linked library version do not match
    VersionMismatch,
    /// Generator was not created using rocrand_create_generator
    NotCreated,
    /// Memory allocation failed during execution
    AllocationFailed,
    /// Generator type is wrong
    TypeError,
    /// Argument out of range
    OutOfRange,
    /// Requested size is not a multiple of quasirandom generator's dimension,
    /// or requested size is not even, or pointer is misaligned
    LengthNotMultiple,
    /// GPU does not have double precision
    DoublePrecisionRequired,
    /// Kernel launch failure
    LaunchFailure,
    /// Internal library error
    InternalError,
    /// Unknown error
    Unknown(u32),
}

/// Specialized Result type for rocrand operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Convert a rocrand status code to a Result
    pub(crate) fn from_status(status: u32) -> Result<()> {
        match status {
            bindings::rocrand_status_ROCRAND_STATUS_SUCCESS => Ok(()),
            bindings::rocrand_status_ROCRAND_STATUS_VERSION_MISMATCH => Err(Error::VersionMismatch),
            bindings::rocrand_status_ROCRAND_STATUS_NOT_CREATED => Err(Error::NotCreated),
            bindings::rocrand_status_ROCRAND_STATUS_ALLOCATION_FAILED => {
                Err(Error::AllocationFailed)
            }
            bindings::rocrand_status_ROCRAND_STATUS_TYPE_ERROR => Err(Error::TypeError),
            bindings::rocrand_status_ROCRAND_STATUS_OUT_OF_RANGE => Err(Error::OutOfRange),
            bindings::rocrand_status_ROCRAND_STATUS_LENGTH_NOT_MULTIPLE => {
                Err(Error::LengthNotMultiple)
            }
            bindings::rocrand_status_ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED => {
                Err(Error::DoublePrecisionRequired)
            }
            bindings::rocrand_status_ROCRAND_STATUS_LAUNCH_FAILURE => Err(Error::LaunchFailure),
            bindings::rocrand_status_ROCRAND_STATUS_INTERNAL_ERROR => Err(Error::InternalError),
            other => Err(Error::Unknown(other)),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::VersionMismatch => {
                write!(f, "Header file and linked library version do not match")
            }
            Error::NotCreated => write!(
                f,
                "Generator was not created using rocrand_create_generator"
            ),
            Error::AllocationFailed => write!(f, "Memory allocation failed during execution"),
            Error::TypeError => write!(f, "Generator type is wrong"),
            Error::OutOfRange => write!(f, "Argument out of range"),
            Error::LengthNotMultiple => write!(
                f,
                "Length not multiple of dimension or other alignment issue"
            ),
            Error::DoublePrecisionRequired => write!(f, "GPU does not have double precision"),
            Error::LaunchFailure => write!(f, "Kernel launch failure"),
            Error::InternalError => write!(f, "Internal library error"),
            Error::Unknown(code) => write!(f, "Unknown error (code: {})", code),
        }
    }
}

impl std::error::Error for Error {}
