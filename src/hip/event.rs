// src/hip/event.rs

use crate::hip::Stream;
use crate::hip::error::{Error, Result};
use crate::hip::ffi;
use std::ptr;

/// Safe wrapper for HIP events
pub struct Event {
    event: ffi::hipEvent_t,
}

impl Event {
    /// Create a new event with default flags
    pub fn new() -> Result<Self> {
        let mut event = ptr::null_mut();
        let error = unsafe { ffi::hipEventCreate(&mut event) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { event })
    }

    /// Create a new event with specific flags
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut event = ptr::null_mut();
        let error = unsafe { ffi::hipEventCreateWithFlags(&mut event, flags) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { event })
    }

    /// Record an event in a stream
    pub fn record(&self, stream: &Stream) -> Result<()> {
        let error = unsafe { ffi::hipEventRecord(self.event, stream.as_raw()) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Synchronize on the event (wait for it to complete)
    pub fn synchronize(&self) -> Result<()> {
        let error = unsafe { ffi::hipEventSynchronize(self.event) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Query if the event has completed
    pub fn query(&self) -> Result<()> {
        let error = unsafe { ffi::hipEventQuery(self.event) };

        if error == ffi::hipError_t_hipSuccess {
            Ok(())
        } else if error == ffi::hipError_t_hipErrorNotReady {
            // Not ready isn't a true error in this context
            Err(Error::new(error))
        } else {
            Err(Error::new(error))
        }
    }

    /// Calculate elapsed time between this event and another in milliseconds
    pub fn elapsed_time(&self, end: &Event) -> Result<f32> {
        let mut time = 0.0;
        let error = unsafe { ffi::hipEventElapsedTime(&mut time, self.event, end.event) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(time)
    }

    /// Get the raw event handle
    pub fn as_raw(&self) -> ffi::hipEvent_t {
        self.event
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                let _ = ffi::hipEventDestroy(self.event);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.event = ptr::null_mut();
        }
    }
}

/// Constants for event creation flags
pub mod event_flags {
    /// Default event creation flag
    pub const DEFAULT: u32 = 0;

    /// Event uses blocking synchronization
    pub const BLOCKING_SYNC: u32 = 1;

    /// Event will not record timing data
    pub const DISABLE_TIMING: u32 = 2;

    /// Event is suitable for interprocess use
    pub const INTERPROCESS: u32 = 4;
}

/// Helper struct to measure elapsed time
pub struct Timer {
    start: Event,
    stop: Event,
}

impl Timer {
    /// Create a new timer
    pub fn new() -> Result<Self> {
        // Create with DISABLE_TIMING = false to enable timing
        let start = Event::new()?;
        let stop = Event::new()?;

        Ok(Self { start, stop })
    }

    /// Start the timer by recording the start event
    pub fn start(&self, stream: &Stream) -> Result<()> {
        self.start.record(stream)
    }

    /// Stop the timer by recording the stop event
    pub fn stop(&self, stream: &Stream) -> Result<()> {
        self.stop.record(stream)
    }

    /// Get the elapsed time in milliseconds
    /// Note: This will synchronize the stop event if it has not completed yet
    pub fn elapsed_time(&self) -> Result<f32> {
        // Make sure the stop event has completed
        self.stop.synchronize()?;

        // Calculate the elapsed time
        self.start.elapsed_time(&self.stop)
    }
}
