// src/rocprofiler/context.rs

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::hip::device::Device;
use crate::hip::stream::Stream; // Your existing Stream type
use crate::rocprofiler::bindings;
use crate::rocprofiler::error::{Error, Result};
use crate::rocprofiler::types::{Feature, Group, ProfilerMode};

/// Handler type for ROCProfiler completion events
pub type Handler = Arc<dyn Fn(Group) -> bool + Send + Sync>;

/// Represents a ROCProfiler context for performance profiling
pub struct Context {
    context: *mut bindings::rocprofiler_t,
    device_id: i32,
    features: Vec<Feature>,
    owned: bool,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Create a new profiling context
    pub fn new(
        device: Device,
        features: Vec<Feature>,
        modes: &[ProfilerMode],
        queue: Option<&Stream>,
        queue_depth: Option<u32>,
        handler: Option<Handler>,
    ) -> Result<Self> {
        let mode = ProfilerMode::combine(modes);

        // Get the device ID for later reference
        let device_id = device.id();

        // Convert device to HSA agent handle for ROCProfiler
        let agent_handle = bindings::hsa_agent_t {
            handle: device_id as u64,
        };

        // Convert features to native format
        let mut feature_handles = Vec::new();
        let mut string_storage = Vec::new();
        let mut param_storage = Vec::new();

        for feature in &features {
            let (native_feature, strings, params) = feature.to_native()?;
            feature_handles.push(native_feature);
            string_storage.push(strings);
            param_storage.push(params);
        }

        // Prepare properties
        let mut properties = bindings::rocprofiler_properties_t {
            queue: if let Some(q) = queue {
                q.as_raw() as *mut crate::rocprofiler::bindings::hsa_queue_t
            } else {
                std::ptr::null_mut()
            },
            queue_depth: queue_depth.unwrap_or(0),
            handler: None,
            handler_arg: std::ptr::null_mut(),
        };

        // TODO: If handler is provided, need to set up a trampoline function
        // This is complex because we need to store the handler somewhere and
        // make it accessible to the C callback

        // Create the context
        let mut context = std::ptr::null_mut();
        let status = unsafe {
            bindings::rocprofiler_open(
                agent_handle,
                if feature_handles.is_empty() {
                    std::ptr::null_mut()
                } else {
                    feature_handles.as_mut_ptr()
                },
                feature_handles.len() as u32,
                &mut context,
                mode,
                &mut properties,
            )
        };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        // Return the context
        Ok(Self {
            context,
            device_id,
            features,
            owned: true,
        })
    }

    /// Create a context from an existing raw context pointer
    pub unsafe fn from_raw(
        context: *mut bindings::rocprofiler_t,
        device_id: i32,
        features: Vec<Feature>,
        owned: bool,
    ) -> Self {
        Self {
            context,
            device_id,
            features,
            owned,
        }
    }

    /// Get the raw context pointer
    pub fn as_raw(&self) -> *mut bindings::rocprofiler_t {
        self.context
    }

    /// Get the device ID associated with this context
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get the device associated with this context
    pub fn device(&self) -> crate::hip::error::Result<Device> {
        Device::new(self.device_id)
    }

    /// Get the features associated with this context
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// Reset the context before reusing
    pub fn reset(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_reset(self.context, group_index) };
        Error::from_rocprofiler_error(status)
    }

    /// Start profiling
    pub fn start(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_start(self.context, group_index) };
        Error::from_rocprofiler_error(status)
    }

    /// Stop profiling
    pub fn stop(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_stop(self.context, group_index) };
        Error::from_rocprofiler_error(status)
    }

    /// Read profiling data
    pub fn read(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_read(self.context, group_index) };
        Error::from_rocprofiler_error(status)
    }

    /// Get profiling data
    pub fn get_data(&self, group_index: u32) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_get_data(self.context, group_index) };
        Error::from_rocprofiler_error(status)
    }

    /// Get the number of profiling groups
    pub fn group_count(&self) -> Result<u32> {
        let mut count = 0;
        let status = unsafe { bindings::rocprofiler_group_count(self.context, &mut count) };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        Ok(count)
    }

    /// Get a profiling group by index
    pub fn get_group(&self, group_index: u32) -> Result<Group> {
        let mut group = unsafe { std::mem::zeroed::<bindings::rocprofiler_group_t>() };

        let status = unsafe {
            bindings::rocprofiler_get_group(self.context, group_index, &mut group)
        };

        if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
            return Err(Error::new(status));
        }

        Ok(Group::from_native(group))
    }

    /// Get metrics data
    pub fn get_metrics(&self) -> Result<()> {
        let status = unsafe { bindings::rocprofiler_get_metrics(self.context) };
        Error::from_rocprofiler_error(status)
    }

    /// Collect profiling data for all features
    pub fn collect_data(&mut self) -> Result<()> {
        // Get the number of groups
        let group_count = self.group_count()?;

        // For each group, get data
        for i in 0..group_count {
            self.get_data(i)?;
        }

        // Get metrics data
        self.get_metrics()?;

        // TODO: Update feature data from the native features
        // This requires accessing the features inside the context

        Ok(())
    }

    /// Iterate trace data with a callback
    pub unsafe fn iterate_trace_data<F>(&self, mut callback: F) -> Result<()>
    where
        F: FnMut(bindings::hsa_ven_amd_aqlprofile_info_type_t, &bindings::hsa_ven_amd_aqlprofile_info_data_t) -> Result<()>,
    {
        unsafe {
            // Create a wrapper for the callback
            extern "C" fn callback_wrapper(
                info_type: bindings::hsa_ven_amd_aqlprofile_info_type_t,
                info_data: *mut bindings::hsa_ven_amd_aqlprofile_info_data_t,
                callback_data: *mut c_void,
            ) -> bindings::hsa_status_t {
                let callback_ptr = callback_data as *mut &mut dyn FnMut(
                    bindings::hsa_ven_amd_aqlprofile_info_type_t,
                    &bindings::hsa_ven_amd_aqlprofile_info_data_t,
                ) -> Result<()>;

                unsafe {
                    match (*callback_ptr)(info_type, &*info_data) {
                        Ok(()) => bindings::hsa_status_t_HSA_STATUS_SUCCESS,
                        Err(e) => e.code(),
                    }
                }
            }

            let mut callback_wrapper_data = &mut callback as &mut dyn FnMut(
                bindings::hsa_ven_amd_aqlprofile_info_type_t,
                &bindings::hsa_ven_amd_aqlprofile_info_data_t,
            ) -> Result<()>;

            let status = bindings::rocprofiler_iterate_trace_data(
                self.context,
                Some(callback_wrapper),
                &mut callback_wrapper_data as *mut _ as *mut c_void,
            );

            Error::from_rocprofiler_error(status)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.owned && !self.context.is_null() {
            unsafe {
                let _ = bindings::rocprofiler_close(self.context);
            }
            self.context = std::ptr::null_mut();
        }
    }
}

/// Properties for profiling context creation
#[derive(Clone)]
pub struct Properties {
    /// Queue for STANDALONE mode
    pub queue: Option<Stream>,
    /// Created queue depth
    pub queue_depth: u32,
    /// Handler on completion
    pub handler: Option<Handler>,
}

impl Properties {
    /// Create new default properties
    pub fn new() -> Self {
        Self {
            queue: None,
            queue_depth: 0,
            handler: None,
        }
    }

    /// Set queue
    pub fn with_queue(mut self, queue: Stream) -> Self {
        self.queue = Some(queue);
        self
    }

    /// Set queue depth
    pub fn with_queue_depth(mut self, depth: u32) -> Self {
        self.queue_depth = depth;
        self
    }

    /// Set completion handler
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(Group) -> bool + Send + Sync + 'static,
    {
        self.handler = Some(Arc::new(handler));
        self
    }
}

impl Default for Properties {
    fn default() -> Self {
        Self::new()
    }
}