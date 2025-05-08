// src/rocprofiler/context.rs

use crate::rocprofiler::error::{Error, Result};
use crate::rocprofiler::types::{Feature, ProfilerMode, Group};
use crate::rocprofiler::bindings;
use crate::hip;
use std::marker::PhantomData;
use std::ptr;
use std::mem;

/// Properties for context creation
pub struct Properties {
    pub(crate) properties: bindings::rocprofiler_properties_t,
    _handler_data: Option<Box<HandlerData>>,
}

// Structure to hold handler data for the C callback
struct HandlerData {
    handler: Box<dyn FnMut(Group) -> bool + Send + 'static>,
}

// The C handler function that will be called by ROCProfiler
unsafe extern "C" fn handler_callback(
    group: bindings::rocprofiler_group_t,
    arg: *mut std::os::raw::c_void,
) -> bool {
    if arg.is_null() {
        return false;
    }

    let handler_data = &mut *(arg as *mut HandlerData);
    let rust_group = Group::from_c(&group);

    // Call the Rust handler
    (handler_data.handler)(rust_group)
}

impl Properties {
    /// Create new default properties
    pub fn new() -> Self {
        Self {
            properties: bindings::rocprofiler_properties_t {
                queue: ptr::null_mut(),
                queue_depth: 0,
                handler: None,
                handler_arg: ptr::null_mut(),
            },
            _handler_data: None,
        }
    }

    /// Set the queue depth
    pub fn with_queue_depth(mut self, depth: u32) -> Self {
        self.properties.queue_depth = depth;
        self
    }

    /// Set a handler function that will be called for each group
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: FnMut(Group) -> bool + Send + 'static,
    {
        // Create the handler data
        let handler_data = Box::new(HandlerData {
            handler: Box::new(handler),
        });

        // Set the handler properties
        self.properties.handler = Some(handler_callback);
        self.properties.handler_arg = Box::into_raw(handler_data) as *mut std::os::raw::c_void;

        // Store the handler data so it's not dropped
        self._handler_data = Some(unsafe { Box::from_raw(self.properties.handler_arg as *mut HandlerData) });

        self
    }

    /// Set the queue directly (advanced usage)
    pub fn with_queue(mut self, queue: *mut hip::ffi::hipStream_t) -> Self {
        self.properties.queue = queue as *mut _;
        self
    }
}

impl Default for Properties {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for profiling
pub struct Context {
    context: *mut bindings::rocprofiler_t,
    features: Vec<Feature>,
    c_features: Vec<*mut bindings::rocprofiler_feature_t>,
    agent: hip::Device,
    _phantom: PhantomData<()>,
}

// Safe to send between threads since ROCProfiler contexts are thread-safe
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Create a new profiling context
    pub fn new(
        device: hip::Device,
        mut features: Vec<Feature>,
        modes: &[ProfilerMode],
        properties: Option<Properties>,
    ) -> Result<Self> {
        // Combine the mode flags
        let mode: u32 = modes.iter().fold(0, |acc, mode| acc | (*mode as u32));

        // Convert features to C representation
        let mut c_features: Vec<*mut bindings::rocprofiler_feature_t> = Vec::with_capacity(features.len());
        for feature in &mut features {
            c_features.push(feature.to_c());
        }

        // Prepare properties
        let properties_ptr = match properties {
            Some(props) => &props.properties as *const _ as *mut _,
            None => ptr::null_mut(),
        };

        // Create the context
        let mut context = ptr::null_mut();

        // Create the HSA agent handle
        let agent = bindings::hsa_agent_t {
            handle: device.id() as u64,
        };

        let status = unsafe {
            bindings::rocprofiler_open(
                agent,
                if c_features.is_empty() { ptr::null_mut() } else { *c_features.as_mut_ptr() },
                c_features.len() as u32,
                &mut context,
                mode,
                properties_ptr,
            )
        };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        Ok(Self {
            context,
            features,
            c_features,
            agent: device,
            _phantom: PhantomData,
        })
    }

    /// Get the number of feature groups
    pub fn group_count(&self) -> Result<u32> {
        let mut count = 0;
        let status = unsafe { bindings::rocprofiler_group_count(self.context, &mut count) };
        Error::from_hsa_status_with_value(status, count)
    }

    /// Get a specific group
    pub fn get_group(&self, index: u32) -> Result<Group> {
        let mut group_data = bindings::rocprofiler_group_t {
            index,
            features: ptr::null_mut(),
            feature_count: 0,
            context: self.context,
        };

        let status = unsafe { bindings::rocprofiler_get_group(self.context, index, &mut group_data) };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        Ok(Group::from_c(&group_data))
    }

    /// Start profiling for a specific group
    pub fn start(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_start(self.context, group_index) };
        Error::from_hsa_status(status)
    }

    /// Stop profiling for a specific group
    pub fn stop(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_stop(self.context, group_index) };
        Error::from_hsa_status(status)
    }

    /// Read profiling data for a specific group
    pub fn read(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_read(self.context, group_index) };
        Error::from_hsa_status(status)
    }

    /// Get profiling data for a specific group
    pub fn get_data(&mut self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_get_data(self.context, group_index) };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        // Update the feature data with the results
        for (i, c_feature) in self.c_features.iter().enumerate() {
            if i < self.features.len() {
                self.features[i].update_from_c(*c_feature);
            }
        }

        Ok(())
    }

    /// Get the features
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Get mutable access to the features
    pub fn features_mut(&mut self) -> &mut [Feature] {
        &mut self.features
    }

    /// Reset a group
    pub fn reset(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_reset(self.context, group_index) };
        Error::from_hsa_status(status)
    }

    /// Get the agent/device
    pub fn agent(&self) -> &hip::Device {
        &self.agent
    }

    /// Get metrics for the current device
    pub fn get_metrics(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_get_metrics(self.context) };
        Error::from_hsa_status(status)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                let _ = bindings::rocprofiler_close(self.context);
                self.context = ptr::null_mut();
            }
        }
    }
}