// src/error.rs
//
// Unified error handling for ROCm-related operations

use std::fmt;

/// A unified error type for ROCm operations that can represent
/// errors from different subsystems (HIP, rocRAND, etc)
#[derive(Debug)]
pub enum Error {
    /// HIP-related error
    Hip(crate::hip::Error),

    /// rocRAND-related error
    RocRand(crate::rocrand::Error),

    /// MIOpen-related error
    MIOpen(crate::miopen::Error),

    RocFFT(crate::rocfft::error::Error),

    /// Custom error with a message
    Custom(String),
}

// Automatic conversion from HIP errors
impl From<crate::hip::Error> for Error {
    fn from(error: crate::hip::Error) -> Self {
        Error::Hip(error)
    }
}

// Automatic conversion from rocRAND errors
impl From<crate::rocrand::Error> for Error {
    fn from(error: crate::rocrand::Error) -> Self {
        Error::RocRand(error)
    }
}

// Automatic conversion from MIOpen errors
impl From<crate::miopen::Error> for Error {
    fn from(error: crate::miopen::Error) -> Self {
        Error::MIOpen(error)
    }
}

impl From<crate::rocfft::error::Error> for Error {
    fn from(error: crate::rocfft::error::Error) -> Self {
        Error::RocFFT(error)
    }
}

// Implement Display for better error messages
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Hip(e) => write!(f, "HIP error: {}", e),
            Error::RocRand(e) => write!(f, "rocRAND error: {}", e),
            Error::MIOpen(e) => write!(f, "MIOpen error: {}", e),
            Error::RocFFT(e) => writeln!(f, "rocFFT error: {}", e),
            Error::Custom(msg) => write!(f, "Error: {}", msg),
        }
    }
}

// Make our error type compatible with the standard error trait
impl std::error::Error for Error {}

/// Specialized Result type for ROCm operations
pub type Result<T> = std::result::Result<T, Error>;

/// Helper function to create a custom error
pub fn custom_error<S: Into<String>>(message: S) -> Error {
    Error::Custom(message.into())
}
