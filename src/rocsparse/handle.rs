//! ROCsparse library context handle

use crate::rocsparse::error::{Result, status_to_result};
use crate::rocsparse::{
    ihipStream_t, rocsparse_create_handle, rocsparse_destroy_handle, rocsparse_get_pointer_mode,
    rocsparse_get_stream, rocsparse_get_version, rocsparse_handle, rocsparse_pointer_mode_,
    rocsparse_pointer_mode__rocsparse_pointer_mode_device,
    rocsparse_pointer_mode__rocsparse_pointer_mode_host, rocsparse_set_pointer_mode,
    rocsparse_set_stream,
};
use std::mem::MaybeUninit;

/// ROCsparse library context
pub struct Handle {
    pub(crate) inner: rocsparse_handle,
}

impl Handle {
    /// Create a new ROCsparse handle
    pub fn new() -> Result<Self> {
        let mut handle = MaybeUninit::uninit();
        let status = unsafe { rocsparse_create_handle(handle.as_mut_ptr()) };
        status_to_result(status)?;
        let handle = unsafe { handle.assume_init() };
        Ok(Self { inner: handle })
    }

    /// Set the stream for the handle
    pub unsafe fn set_stream(&self, stream: *mut ihipStream_t) -> Result<()> {
        let status = unsafe { rocsparse_set_stream(self.inner, stream) };
        status_to_result(status)
    }

    /// Get the current stream
    pub fn get_stream(&self) -> Result<*mut ihipStream_t> {
        let mut stream = MaybeUninit::uninit();
        let status = unsafe { rocsparse_get_stream(self.inner, stream.as_mut_ptr()) };
        status_to_result(status)?;
        Ok(unsafe { stream.assume_init() })
    }

    /// Set pointer mode
    pub fn set_pointer_mode(&self, mode: PointerMode) -> Result<()> {
        let status = unsafe { rocsparse_set_pointer_mode(self.inner, mode.into()) };
        status_to_result(status)
    }

    /// Get pointer mode
    pub fn get_pointer_mode(&self) -> Result<PointerMode> {
        let mut mode = MaybeUninit::uninit();
        let status = unsafe { rocsparse_get_pointer_mode(self.inner, mode.as_mut_ptr()) };
        status_to_result(status)?;
        Ok(unsafe { PointerMode::from_raw(mode.assume_init()) })
    }

    /// Get ROCsparse version
    pub fn get_version(&self) -> Result<(u32, u32, u32)> {
        let mut version = MaybeUninit::uninit();
        let status = unsafe { rocsparse_get_version(self.inner, version.as_mut_ptr()) };
        status_to_result(status)?;
        let version = unsafe { version.assume_init() };
        let patch = version % 100;
        let minor = (version / 100) % 1000;
        let major = version / 100000;
        Ok((major as u32, minor as u32, patch as u32))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_handle(self.inner);
        }
    }
}

/// Pointer mode for ROCsparse functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerMode {
    /// Scalar pointers are in host memory
    Host,
    /// Scalar pointers are in device memory
    Device,
}

impl PointerMode {
    pub(crate) fn from_raw(raw: rocsparse_pointer_mode_) -> Self {
        match raw {
            rocsparse_pointer_mode__rocsparse_pointer_mode_device => PointerMode::Device,
            _ => PointerMode::Host,
        }
    }
}

impl From<PointerMode> for rocsparse_pointer_mode_ {
    fn from(mode: PointerMode) -> Self {
        match mode {
            PointerMode::Host => rocsparse_pointer_mode__rocsparse_pointer_mode_host,
            PointerMode::Device => rocsparse_pointer_mode__rocsparse_pointer_mode_device,
        }
    }
}
