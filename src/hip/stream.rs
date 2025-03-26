// src/hip/stream.rs

use std::ptr;
use crate::hip;
use crate::hip::ffi;
use crate::hip::error::{Error, Result};
use crate::hip::event::Event;

/// Safe wrapper for HIP streams
pub struct Stream {
    stream: hip::ffi::hipStream_t
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a new stream
    pub fn new() -> Result<Self> {
        let mut stream = ptr::null_mut();
        let error = unsafe { ffi::hipStreamCreate(&mut stream) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { stream })
    }

    /// Create a new stream with specific flags
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut stream = ptr::null_mut();
        let error = unsafe { ffi::hipStreamCreateWithFlags(&mut stream, flags) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { stream })
    }

    /// Create a new stream with priority
    pub fn with_priority(flags: u32, priority: i32) -> Result<Self> {
        let mut stream = ptr::null_mut();
        let error = unsafe { ffi::hipStreamCreateWithPriority(&mut stream, flags, priority) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { stream })
    }

    /// Wait for a stream to complete
    pub fn synchronize(&self) -> Result<()> {
        let error = unsafe { ffi::hipStreamSynchronize(self.stream) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Query if all operations in the stream have completed
    pub fn query(&self) -> Result<()> {
        let error = unsafe { ffi::hipStreamQuery(self.stream) };

        if error == ffi::hipError_t_hipSuccess {
            Ok(())
        } else if error == ffi::hipError_t_hipErrorNotReady {
            // Not ready isn't a true error in this context
            Err(Error::new(error))
        } else {
            Err(Error::new(error))
        }
    }

    /// Wait on an event
    pub fn wait_event(&self, event: &Event, flags: u32) -> Result<()> {
        let error = unsafe {
            ffi::hipStreamWaitEvent(self.stream, event.as_raw(), flags)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Add a callback to be executed when the stream completes
    pub fn add_callback<F>(&self, callback: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        // Create a Box for our closure
        let callback_box = Box::new(callback);

        // Convert the Box to a raw pointer (to pass through the C API)
        let callback_ptr = Box::into_raw(callback_box);

        // The C callback function that will be called by HIP
        unsafe extern "C" fn helper_callback(
            _stream: ffi::hipStream_t,
            _status: ffi::hipError_t,
            user_data: *mut std::ffi::c_void
        ) {
            // Convert the pointer back to a Box and then to our closure
            let callback_box = Box::from_raw(user_data as *mut Box<dyn FnOnce()>);

            // Call the closure
            (*callback_box)();
        }

        let error = unsafe {
            ffi::hipStreamAddCallback(
                self.stream,
                Some(helper_callback),
                callback_ptr as *mut std::ffi::c_void,
                0
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            // If there was an error, we need to clean up our Box to avoid memory leak
            unsafe {
                let _ = Box::from_raw(callback_ptr);
            }
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Get the raw stream handle
    pub fn as_raw(&self) -> ffi::hipStream_t {
        self.stream
    }

    /// Get the stream priority range
    pub fn priority_range() -> Result<(i32, i32)> {
        let mut least_priority = 0;
        let mut greatest_priority = 0;

        let error = unsafe {
            ffi::hipDeviceGetStreamPriorityRange(&mut least_priority, &mut greatest_priority)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok((least_priority, greatest_priority))
    }

    /// Get the priority of this stream
    pub fn get_priority(&self) -> Result<i32> {
        let mut priority = 0;

        let error = unsafe {
            ffi::hipStreamGetPriority(self.stream, &mut priority)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(priority)
    }

    /// Get the flags of this stream
    pub fn get_flags(&self) -> Result<u32> {
        let mut flags = 0;

        let error = unsafe {
            ffi::hipStreamGetFlags(self.stream, &mut flags)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(flags)
    }

    /// Get the device associated with this stream
    pub fn get_device(&self) -> Result<i32> {
        let mut device = 0;

        let error = unsafe {
            ffi::hipStreamGetDevice(self.stream, &mut device)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(device)
    }
    pub fn from_raw(stream: ffi::hipStream_t) -> Self {
        Self { stream }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = ffi::hipStreamDestroy(self.stream);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.stream = ptr::null_mut();
        }
    }
}

/// Constants for stream creation flags
pub mod stream_flags {
    /// Default stream creation flag (synchronizing)
    pub const DEFAULT: u32 = 0;

    /// Non-blocking stream that doesn't synchronize with the NULL stream
    pub const NON_BLOCKING: u32 = 1;
}