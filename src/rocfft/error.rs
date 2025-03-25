use std::fmt;
use std::error::Error as StdError;
use std::ffi::NulError;
use crate::rocfft::bindings;

/// Custom error type for rocFFT operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A general rocFFT operation failed
    Failure,
    /// An invalid argument value was provided
    InvalidArgValue,
    /// Invalid transform dimensions were specified
    InvalidDimensions,
    /// An invalid array type was specified
    InvalidArrayType,
    /// Invalid stride values were specified
    InvalidStrides,
    /// An invalid distance value was specified
    InvalidDistance,
    /// An invalid offset value was specified
    InvalidOffset,
    /// An invalid work buffer was provided
    InvalidWorkBuffer,
    /// A null pointer was encountered where a valid pointer was required
    NullPointer,
    /// A non-UTF8 string was encountered
    InvalidString,
    /// An operation was attempted on an object that has already been destroyed
    ObjectDestroyed,
    /// Input and output data types are incompatible for the requested transform
    IncompatibleTypes,
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
            Error::Failure => write!(f, "rocFFT operation failed"),
            Error::InvalidArgValue => write!(f, "Invalid argument value"),
            Error::InvalidDimensions => write!(f, "Invalid dimensions"),
            Error::InvalidArrayType => write!(f, "Invalid array type"),
            Error::InvalidStrides => write!(f, "Invalid strides"),
            Error::InvalidDistance => write!(f, "Invalid distance"),
            Error::InvalidOffset => write!(f, "Invalid offset"),
            Error::InvalidWorkBuffer => write!(f, "Invalid work buffer"),
            Error::NullPointer => write!(f, "Null pointer"),
            Error::InvalidString => write!(f, "Invalid string"),
            Error::ObjectDestroyed => write!(f, "Object has been destroyed"),
            Error::IncompatibleTypes => write!(f, "Incompatible input/output data types"),
            Error::OutOfMemory => write!(f, "Out of memory"),
            Error::NulError(msg) => write!(f, "C string conversion error: {}", msg),
            Error::InvalidDevice => write!(f, "Invalid device or device context"),
            Error::UnsupportedConfiguration => write!(f, "Unsupported configuration of parameters"),
            Error::Unknown(code) => write!(f, "Unknown rocFFT error (code: {})", code),
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
            bindings::rocfft_status_e_rocfft_status_success => panic!("Tried to convert successful status to error"),
            bindings::rocfft_status_e_rocfft_status_failure => Error::Failure,
            bindings::rocfft_status_e_rocfft_status_invalid_arg_value => Error::InvalidArgValue,
            bindings::rocfft_status_e_rocfft_status_invalid_dimensions => Error::InvalidDimensions,
            bindings::rocfft_status_e_rocfft_status_invalid_array_type => Error::InvalidArrayType,
            bindings::rocfft_status_e_rocfft_status_invalid_strides => Error::InvalidStrides,
            bindings::rocfft_status_e_rocfft_status_invalid_distance => Error::InvalidDistance,
            bindings::rocfft_status_e_rocfft_status_invalid_offset => Error::InvalidOffset,
            bindings::rocfft_status_e_rocfft_status_invalid_work_buffer => Error::InvalidWorkBuffer,
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
            "incompatible_types" => Error::IncompatibleTypes,
            "out_of_memory" => Error::OutOfMemory,
            "invalid_device" => Error::InvalidDevice,
            "unsupported_configuration" => Error::UnsupportedConfiguration,
            _ => Error::Failure,
        }
    }
}

/// Custom Result type for rocFFT operations
pub type Result<T> = std::result::Result<T, Error>;

/// Check a rocFFT status code and convert to a Rust Result
pub(crate) fn check_error(status: u32) -> Result<()> {
    match status {
        bindings::rocfft_status_e_rocfft_status_success => Ok(()),
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

/// Validate transform dimensions (must be 1, 2, or 3)
#[inline]
pub(crate) fn check_dimensions(dimensions: usize) -> Result<()> {
    if dimensions < 1 || dimensions > 3 {
        Err(Error::InvalidDimensions)
    } else {
        Ok(())
    }
}

/// Helper function to detect incompatible array types for a transform
#[inline]
pub(crate) fn check_compatible_types(
    transform_type: u32,
    in_array_type: u32,
    out_array_type: u32
) -> Result<()> {
    match transform_type {
        bindings::rocfft_transform_type_e_rocfft_transform_type_complex_forward |
        bindings::rocfft_transform_type_e_rocfft_transform_type_complex_inverse => {
            // Both input and output must be complex
            if (in_array_type == bindings::rocfft_array_type_e_rocfft_array_type_complex_interleaved ||
                in_array_type == bindings::rocfft_array_type_e_rocfft_array_type_complex_planar) &&
                (out_array_type == bindings::rocfft_array_type_e_rocfft_array_type_complex_interleaved ||
                    out_array_type == bindings::rocfft_array_type_e_rocfft_array_type_complex_planar) {
                Ok(())
            } else {
                Err(Error::IncompatibleTypes)
            }
        },
        bindings::rocfft_transform_type_e_rocfft_transform_type_real_forward => {
            // Input must be real, output must be hermitian
            if in_array_type == bindings::rocfft_array_type_e_rocfft_array_type_real &&
                (out_array_type == bindings::rocfft_array_type_e_rocfft_array_type_hermitian_interleaved ||
                    out_array_type == bindings::rocfft_array_type_e_rocfft_array_type_hermitian_planar) {
                Ok(())
            } else {
                Err(Error::IncompatibleTypes)
            }
        },
        bindings::rocfft_transform_type_e_rocfft_transform_type_real_inverse => {
            // Input must be hermitian, output must be real
            if (in_array_type == bindings::rocfft_array_type_e_rocfft_array_type_hermitian_interleaved ||
                in_array_type == bindings::rocfft_array_type_e_rocfft_array_type_hermitian_planar) &&
                out_array_type == bindings::rocfft_array_type_e_rocfft_array_type_real {
                Ok(())
            } else {
                Err(Error::IncompatibleTypes)
            }
        },
        _ => Err(Error::InvalidArgValue),
    }
}