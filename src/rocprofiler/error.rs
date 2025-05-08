// src/rocprofiler/error.rs

use std::fmt;
use std::error::Error as StdError;

use crate::rocprofiler::bindings;

/// Error type for ROCProfiler operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    pub(crate) code: bindings::hsa_status_t,
}

/// Result type for ROCProfiler operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a new error from a ROCProfiler error code
    pub fn new(code: bindings::hsa_status_t) -> Self {
        Self { code }
    }

    /// Get the raw error code
    pub fn code(&self) -> bindings::hsa_status_t {
        self.code
    }

    /// Returns true if the error code represents success
    pub fn is_success(&self) -> bool {
        self.code == bindings::hsa_status_t_HSA_STATUS_SUCCESS
    }

    /// Convert a ROCProfiler error code to a Result
    pub fn from_rocprofiler_error<T>(error: bindings::hsa_status_t) -> Result<T>
    where
        T: Default,
    {
        if error == bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            Ok(T::default())
        } else {
            Err(Error::new(error))
        }
    }

    /// Convert a ROCProfiler error code to a Result with a specific value
    pub fn from_rocprofiler_error_with_value<T>(
        error: bindings::hsa_status_t,
        value: T,
    ) -> Result<T> {
        if error == bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            Ok(value)
        } else {
            Err(Error::new(error))
        }
    }

    /// Get a human-readable error description
    pub fn description(&self) -> &'static str {
        match self.code {
            bindings::hsa_status_t_HSA_STATUS_SUCCESS => "Success",
            bindings::hsa_status_t_HSA_STATUS_INFO_BREAK => "Break was requested in callback",
            bindings::hsa_status_t_HSA_STATUS_ERROR => "Generic error",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT => "Invalid argument",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_QUEUE_CREATION => "Invalid queue creation",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ALLOCATION => "Invalid allocation",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_AGENT => "Invalid agent",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_REGION => "Invalid region",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_SIGNAL => "Invalid signal",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_QUEUE => "Invalid queue",
            bindings::hsa_status_t_HSA_STATUS_ERROR_OUT_OF_RESOURCES => "Out of resources",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_PACKET_FORMAT => "Invalid packet format",
            bindings::hsa_status_t_HSA_STATUS_ERROR_RESOURCE_FREE => "Error freeing resources",
            bindings::hsa_status_t_HSA_STATUS_ERROR_NOT_INITIALIZED => "Not initialized",
            bindings::hsa_status_t_HSA_STATUS_ERROR_REFCOUNT_OVERFLOW => "Reference count overflow",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS => "Incompatible arguments",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_INDEX => "Invalid index",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ISA => "Invalid ISA",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ISA_NAME => "Invalid ISA name",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_CODE_OBJECT => "Invalid code object",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_EXECUTABLE => "Invalid executable",
            bindings::hsa_status_t_HSA_STATUS_ERROR_FROZEN_EXECUTABLE => "Executable is frozen",
            bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_SYMBOL_NAME => "Invalid symbol name",
            bindings::hsa_status_t_HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED => "Variable already defined",
            bindings::hsa_status_t_HSA_STATUS_ERROR_VARIABLE_UNDEFINED => "Variable undefined",
            bindings::hsa_status_t_HSA_STATUS_ERROR_EXCEPTION => "HSAIL operation resulted in exception",
            _ => "Unknown error",
        }
    }

    /// Get an error string from the ROCProfiler library
    pub fn error_string() -> Result<String> {
        let mut str_ptr: *const ::std::os::raw::c_char = std::ptr::null();
        let status = unsafe { bindings::rocprofiler_error_string(&mut str_ptr) };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        if str_ptr.is_null() {
            return Ok("No error string available".to_string());
        }

        let c_str = unsafe { std::ffi::CStr::from_ptr(str_ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ROCProfiler error {}: {}",
            self.code,
            self.description()
        )
    }
}

impl StdError for Error {}