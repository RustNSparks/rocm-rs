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

    #[cfg(feature = "miopen")]
    /// MIOpen-related error (if you have this module)
    MIOpen(crate::miopen::Error),

    /// rocFFT-related error (if you have this module)
    RocFFT(crate::rocfft::error::Error),

    /// rocBLAS-related error (if you have this module)
    RocBLAS(crate::rocblas::Error),
    /// Custom error with a message
    Custom(String),

    /// Invalid operation error
    InvalidOperation(String),

    /// Memory allocation error
    OutOfMemory(String),

    /// Invalid argument error
    InvalidArgument(String),

    /// Not implemented error
    NotImplemented(String),

    /// I/O related error
    Io(std::io::Error),

    /// Parsing error
    Parse(String),

    /// Timeout error
    Timeout(String),

    /// Device error
    DeviceError(String),

    /// Kernel compilation error
    KernelCompilation(String),

    /// Synchronization error
    SynchronizationError(String),
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

// Automatic conversion from MIOpen errors (if feature is enabled)
#[cfg(feature = "miopen")]
impl From<crate::miopen::Error> for Error {
    fn from(error: crate::miopen::Error) -> Self {
        Error::MIOpen(error)
    }
}

// Automatic conversion from rocFFT errors (if feature is enabled)
impl From<crate::rocfft::error::Error> for Error {
    fn from(error: crate::rocfft::error::Error) -> Self {
        Error::RocFFT(error)
    }
}

// Automatic conversion from I/O errors
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

// Implement Display for better error messages
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Hip(e) => write!(f, "HIP error: {}", e),
            Error::RocRand(e) => write!(f, "rocRAND error: {}", e),
            #[cfg(feature = "miopen")]
            Error::MIOpen(e) => write!(f, "MIOpen error: {}", e),
            Error::RocFFT(e) => write!(f, "rocFFT error: {}", e),
            Error::RocBLAS(e) => write!(f, "rocBLAS error: {}", e),
            Error::Custom(msg) => write!(f, "Error: {}", msg),
            Error::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            Error::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            Error::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            Error::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            Error::Io(e) => write!(f, "I/O error: {}", e),
            Error::Parse(msg) => write!(f, "Parse error: {}", msg),
            Error::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Error::DeviceError(msg) => write!(f, "Device error: {}", msg),
            Error::KernelCompilation(msg) => write!(f, "Kernel compilation error: {}", msg),
            Error::SynchronizationError(msg) => write!(f, "Synchronization error: {}", msg),
        }
    }
}

// Make our error type compatible with the standard error trait
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Hip(e) => Some(e),
            Error::RocRand(e) => Some(e),
            #[cfg(feature = "miopen")]
            Error::MIOpen(e) => Some(e),
            Error::RocFFT(e) => Some(e),
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Specialized Result type for ROCm operations
pub type Result<T> = std::result::Result<T, Error>;

/// Helper function to create a custom error
pub fn custom_error<S: Into<String>>(message: S) -> Error {
    Error::Custom(message.into())
}

/// Helper function to create an invalid operation error
pub fn invalid_operation<S: Into<String>>(message: S) -> Error {
    Error::InvalidOperation(message.into())
}

/// Helper function to create an out of memory error
pub fn out_of_memory<S: Into<String>>(message: S) -> Error {
    Error::OutOfMemory(message.into())
}

/// Helper function to create an invalid argument error
pub fn invalid_argument<S: Into<String>>(message: S) -> Error {
    Error::InvalidArgument(message.into())
}

/// Helper function to create a not implemented error
pub fn not_implemented<S: Into<String>>(message: S) -> Error {
    Error::NotImplemented(message.into())
}

/// Helper function to create a parse error
pub fn parse_error<S: Into<String>>(message: S) -> Error {
    Error::Parse(message.into())
}

/// Helper function to create a timeout error
pub fn timeout_error<S: Into<String>>(message: S) -> Error {
    Error::Timeout(message.into())
}

/// Helper function to create a device error
pub fn device_error<S: Into<String>>(message: S) -> Error {
    Error::DeviceError(message.into())
}

/// Helper function to create a kernel compilation error
pub fn kernel_compilation_error<S: Into<String>>(message: S) -> Error {
    Error::KernelCompilation(message.into())
}

/// Helper function to create a synchronization error
pub fn synchronization_error<S: Into<String>>(message: S) -> Error {
    Error::SynchronizationError(message.into())
}

/// Macro for creating custom errors with formatted messages
#[macro_export]
macro_rules! rocm_error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::Error::$kind(format!($($arg)*))
    };
}

/// Macro for creating custom errors with formatted messages
#[macro_export]
macro_rules! custom_error {
    ($($arg:tt)*) => {
        $crate::error::custom_error(format!($($arg)*))
    };
}

/// Macro for creating invalid operation errors with formatted messages
#[macro_export]
macro_rules! invalid_operation {
    ($($arg:tt)*) => {
        $crate::error::invalid_operation(format!($($arg)*))
    };
}

/// Context trait for adding context to errors
pub trait ErrorContext<T> {
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    fn context<S: Into<String>>(self, msg: S) -> Result<T>;
}

impl<T> ErrorContext<T> for Result<T> {
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| Error::Custom(format!("{}: {}", f(), e)))
    }

    fn context<S: Into<String>>(self, msg: S) -> Result<T> {
        self.map_err(|e| Error::Custom(format!("{}: {}", msg.into(), e)))
    }
}
/// Error code constants for common error types
pub mod error_codes {
    /// Success
    pub const SUCCESS: i32 = 0;

    /// Generic error
    pub const ERROR: i32 = -1;

    /// Invalid argument
    pub const INVALID_ARGUMENT: i32 = -2;

    /// Out of memory
    pub const OUT_OF_MEMORY: i32 = -3;

    /// Not implemented
    pub const NOT_IMPLEMENTED: i32 = -4;

    /// Device error
    pub const DEVICE_ERROR: i32 = -5;

    /// Kernel compilation error
    pub const KERNEL_COMPILATION: i32 = -6;

    /// Synchronization error
    pub const SYNCHRONIZATION_ERROR: i32 = -7;

    /// Timeout
    pub const TIMEOUT: i32 = -8;

    /// I/O error
    pub const IO_ERROR: i32 = -9;

    /// Parse error
    pub const PARSE_ERROR: i32 = -10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = custom_error("Test error");
        assert_eq!(format!("{}", err), "Error: Test error");

        let err = invalid_operation("Invalid operation test");
        assert_eq!(
            format!("{}", err),
            "Invalid operation: Invalid operation test"
        );
    }

    #[test]
    fn test_error_macros() {
        let err = rocm_error!(InvalidOperation, "Test {} error", "formatted");
        match err {
            Error::InvalidOperation(msg) => assert_eq!(msg, "Test formatted error"),
            _ => panic!("Expected InvalidOperation error"),
        }
    }
}
