// src/rocprofiler/profiler.rs

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

use std::sync::RwLock;
use once_cell::sync::Lazy;

use crate::hip::device::Device;
use crate::hip::stream::Stream;
use crate::rocprofiler::bindings;
use crate::rocprofiler::context::{Context, Properties};
use crate::rocprofiler::error::{Error, Result};
use crate::rocprofiler::types::{Feature, InfoData, InfoKind, ProfilerMode};

/// Main ROCProfiler interface for performance profiling
pub struct Profiler {
    context: Context,
}

impl Profiler {
    /// Create a new profiler for the specified device with the given features
    pub fn new(
        device: Device,
        features: Vec<Feature>,
        modes: &[ProfilerMode],
        properties: Option<Properties>,
    ) -> Result<Self> {
        // Create the context
        let context = if let Some(props) = properties {
            Context::new(
                device,
                features,
                modes,
                props.queue.as_ref(),
                Some(props.queue_depth),
                props.handler,
            )?
        } else {
            Context::new(device, features, modes, None, None, None)?
        };

        Ok(Self { context })
    }

    /// Get the underlying context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get the features being profiled
    pub fn features(&self) -> &[Feature] {
        self.context.features()
    }

    /// Start profiling
    pub fn start(&self, group_index: u32) -> Result<()> {
        self.context.start(group_index)
    }

    /// Stop profiling
    pub fn stop(&self, group_index: u32) -> Result<()> {
        self.context.stop(group_index)
    }

    /// Read profiling data
    pub fn read(&self, group_index: u32) -> Result<()> {
        self.context.read(group_index)
    }

    /// Get profiling data
    pub fn get_data(&self, group_index: u32) -> Result<()> {
        self.context.get_data(group_index)
    }

    /// Collect profiling data for all features
    pub fn collect_data(&mut self) -> Result<()> {
        self.context.collect_data()
    }

    /// Run a complete profiling session for a single group
    pub fn profile_group(&mut self, group_index: u32) -> Result<()> {
        // Start profiling
        self.start(group_index)?;

        // Stop profiling
        self.stop(group_index)?;

        // Read profiling data
        self.read(group_index)?;

        // Get profiling data
        self.get_data(group_index)?;

        Ok(())
    }

    /// Run a complete profiling session for all groups
    pub fn profile_all(&mut self) -> Result<()> {
        // Get the number of groups
        let group_count = self.context.group_count()?;

        // Profile each group
        for i in 0..group_count {
            self.profile_group(i)?;
        }

        // Collect all data
        self.collect_data()?;

        Ok(())
    }
}

/// Create a profiled queue for ROCProfiler
pub fn create_profiled_queue(
    device: Device,
    size: u32,
    private_segment_size: u32,
    group_segment_size: u32,
) -> Result<Stream> {
    let mut queue_ptr = ptr::null_mut();

    // Create an HSA agent from the device ID for ROCProfiler
    let agent_handle = bindings::hsa_agent_t {
        handle: device.id() as u64
    };

    let status = unsafe {
        bindings::rocprofiler_queue_create_profiled(
            agent_handle,
            size,
            0, // Queue type (default)
            None, // Callback
            ptr::null_mut(), // Callback data
            private_segment_size,
            group_segment_size,
            &mut queue_ptr,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    // Create a Stream wrapper from the raw handle
    unsafe { Ok(Stream::from_raw(queue_ptr as crate::hip::bindings::hipStream_t)) }
}

/// Get ROCProfiler version as a tuple of (major, minor)
pub fn version() -> (u32, u32) {
    let major = unsafe { bindings::rocprofiler_version_major() };
    let minor = unsafe { bindings::rocprofiler_version_minor() };
    (major, minor)
}

/// Get ROCProfiler version as a formatted string
pub fn version_string() -> String {
    let (major, minor) = version();
    format!("{}.{}", major, minor)
}

/// Get error string from ROCProfiler
pub fn error_string() -> Result<String> {
    let mut str_ptr: *const ::std::os::raw::c_char = ptr::null();
    let status = unsafe { bindings::rocprofiler_error_string(&mut str_ptr) };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    if str_ptr.is_null() {
        return Ok("No error string available".to_string());
    }

    let c_str = unsafe { CStr::from_ptr(str_ptr) };
    Ok(c_str.to_string_lossy().into_owned())
}

/// Get info for a specific info kind
pub fn get_info(device: Option<&Device>, kind: InfoKind) -> Result<Vec<InfoData>> {
    let mut result = Vec::new();

    // Callback function to collect info data
    extern "C" fn callback(
        info: bindings::rocprofiler_info_data_t,
        data: *mut ::std::ffi::c_void,
    ) -> bindings::hsa_status_t {
        let result_ptr = data as *mut Vec<InfoData>;
        let info_data = unsafe { InfoData::from_native(&info) };

        unsafe {
            (*result_ptr).push(info_data);
        }

        bindings::hsa_status_t_HSA_STATUS_SUCCESS
    }

    // Convert device to agent handle for ROCProfiler
    let agent_handle = device.map(|d| bindings::hsa_agent_t {
            handle: d.id() as u64
        });

    // Call the ROCProfiler API
    let status = unsafe {
        bindings::rocprofiler_iterate_info(
            if let Some(a) = agent_handle {
                &a
            } else {
                ptr::null()
            },
            kind.to_native(),
            Some(callback),
            &mut result as *mut _ as *mut ::std::ffi::c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(result)
}

/// Get all available metrics for a specific device
pub fn get_metrics(device: Option<&Device>) -> Result<Vec<InfoData>> {
    get_info(device, InfoKind::Metric)
}

/// Get the number of available metrics for a specific device
pub fn get_metric_count(device: Option<&Device>) -> Result<u32> {
    let mut count = 0;

    // Convert device to agent handle for ROCProfiler
    let agent_handle = device.map(|d| bindings::hsa_agent_t {
            handle: d.id() as u64
        });

    // Call the ROCProfiler API
    let status = unsafe {
        bindings::rocprofiler_get_info(
            if let Some(a) = agent_handle {
                &a
            } else {
                ptr::null()
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_METRIC_COUNT,
            &mut count as *mut _ as *mut ::std::ffi::c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(count)
}

/// Get all available traces for a specific device
pub fn get_traces(device: Option<&Device>) -> Result<Vec<InfoData>> {
    get_info(device, InfoKind::Trace)
}

/// Get the number of available traces for a specific device
pub fn get_trace_count(device: Option<&Device>) -> Result<u32> {
    let mut count = 0;

    // Convert device to agent handle for ROCProfiler
    let agent_handle = device.map(|d| bindings::hsa_agent_t {
            handle: d.id() as u64
        });

    // Call the ROCProfiler API
    let status = unsafe {
        bindings::rocprofiler_get_info(
            if let Some(a) = agent_handle {
                &a
            } else {
                ptr::null()
            },
            bindings::rocprofiler_info_kind_t_ROCPROFILER_INFO_KIND_TRACE_COUNT,
            &mut count as *mut _ as *mut ::std::ffi::c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(count)
}

/// Get all available trace parameters for a specific device
pub fn get_trace_parameters(device: Option<&Device>, trace_name: &str) -> Result<Vec<InfoData>> {
    let mut result = Vec::new();

    // Callback function to collect info data
    extern "C" fn callback(
        info: bindings::rocprofiler_info_data_t,
        data: *mut ::std::ffi::c_void,
    ) -> bindings::hsa_status_t {
        let result_ptr = data as *mut Vec<InfoData>;
        let info_data = unsafe { InfoData::from_native(&info) };

        unsafe {
            (*result_ptr).push(info_data);
        }

        bindings::hsa_status_t_HSA_STATUS_SUCCESS
    }

    // Convert device to agent handle for ROCProfiler
    let agent_handle = device.map(|d| bindings::hsa_agent_t {
            handle: d.id() as u64
        });

    // Create trace name C string
    let trace_name_cstr = CString::new(trace_name)
        .map_err(|_| Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT))?;

    // Create query
    let query = bindings::rocprofiler_info_query_t {
        trace_parameter: bindings::rocprofiler_info_query_t__bindgen_ty_1 {
            trace_name: trace_name_cstr.as_ptr(),
        },
    };

    // Call the ROCProfiler API
    let status = unsafe {
        bindings::rocprofiler_query_info(
            if let Some(a) = agent_handle {
                &a
            } else {
                ptr::null()
            },
            query,
            Some(callback),
            &mut result as *mut _ as *mut ::std::ffi::c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(result)
}

/// Enable queue callbacks for profiling
pub struct QueueCallbacks {
    dispatch: Option<Box<dyn Fn(&bindings::rocprofiler_callback_data_t, &mut bindings::rocprofiler_group_t) -> Result<()> + Send + Sync + 'static>>,
    create: Option<Box<dyn Fn(*mut bindings::hsa_queue_t) -> Result<()> + Send + Sync + 'static>>,
    destroy: Option<Box<dyn Fn(*mut bindings::hsa_queue_t) -> Result<()> + Send + Sync + 'static>>,
    data: Arc<()>, // Placeholder for user data
}

impl QueueCallbacks {
    /// Create new empty callbacks
    pub fn new() -> Self {
        Self {
            dispatch: None,
            create: None,
            destroy: None,
            data: Arc::new(()),
        }
    }

    /// Set dispatch callback
    pub fn with_dispatch<F>(mut self, callback: F) -> Self
    where
        F: Fn(&bindings::rocprofiler_callback_data_t, &mut bindings::rocprofiler_group_t) -> Result<()> + Send + Sync + 'static,
    {
        self.dispatch = Some(Box::new(callback));
        self
    }

    /// Set create callback
    pub fn with_create<F>(mut self, callback: F) -> Self
    where
        F: Fn(*mut bindings::hsa_queue_t) -> Result<()> + Send + Sync + 'static,
    {
        self.create = Some(Box::new(callback));
        self
    }

    /// Set destroy callback
    pub fn with_destroy<F>(mut self, callback: F) -> Self
    where
        F: Fn(*mut bindings::hsa_queue_t) -> Result<()> + Send + Sync + 'static,
    {
        self.destroy = Some(Box::new(callback));
        self
    }
}

// Global storage for callbacks

// Trampoline functions for C callbacks
static QUEUE_CALLBACKS: Lazy<RwLock<Option<QueueCallbacks>>> = Lazy::new(|| RwLock::new(None));

// Trampoline functions for C callbacks
extern "C" fn dispatch_callback(
    callback_data: *const bindings::rocprofiler_callback_data_t,
    user_data: *mut ::std::ffi::c_void,
    group: *mut bindings::rocprofiler_group_t,
) -> bindings::hsa_status_t {
    // Try to acquire a read lock on the callbacks
    if let Ok(callbacks_guard) = QUEUE_CALLBACKS.read() {
        if let Some(callbacks) = &*callbacks_guard {
            if let Some(dispatch) = &callbacks.dispatch {
                // Safety: We're trusting that callback_data and group are valid pointers
                // from the ROCProfiler library
                let result = unsafe { dispatch(&*callback_data, &mut *group) };
                match result {
                    Ok(()) => return bindings::hsa_status_t_HSA_STATUS_SUCCESS,
                    Err(e) => return e.code(),
                }
            }
        }
    }

    // Default return if anything fails
    bindings::hsa_status_t_HSA_STATUS_SUCCESS
}

extern "C" fn create_callback(
    queue: *mut bindings::hsa_queue_t,
    data: *mut ::std::ffi::c_void,
) -> bindings::hsa_status_t {
    // Try to acquire a read lock on the callbacks
    if let Ok(callbacks_guard) = QUEUE_CALLBACKS.read() {
        if let Some(callbacks) = &*callbacks_guard {
            if let Some(create) = &callbacks.create {
                let result = unsafe { create(queue) };
                match result {
                    Ok(()) => return bindings::hsa_status_t_HSA_STATUS_SUCCESS,
                    Err(e) => return e.code(),
                }
            }
        }
    }

    // Default return if anything fails
    bindings::hsa_status_t_HSA_STATUS_SUCCESS
}

extern "C" fn destroy_callback(
    queue: *mut bindings::hsa_queue_t,
    data: *mut ::std::ffi::c_void,
) -> bindings::hsa_status_t {
    // Try to acquire a read lock on the callbacks
    if let Ok(callbacks_guard) = QUEUE_CALLBACKS.read() {
        if let Some(callbacks) = &*callbacks_guard {
            if let Some(destroy) = &callbacks.destroy {
                let result = unsafe { destroy(queue) };
                return match result {
                    Ok(()) => bindings::hsa_status_t_HSA_STATUS_SUCCESS,
                    Err(e) => e.code(),
                }
            }
        }
    }

    // Default return if anything fails
    bindings::hsa_status_t_HSA_STATUS_SUCCESS
}

/// Set queue callbacks for profiling
pub fn set_queue_callbacks(callbacks: QueueCallbacks) -> Result<()> {
    let native_callbacks = bindings::rocprofiler_queue_callbacks_t {
        dispatch: Some(dispatch_callback),
        create: Some(create_callback),
        destroy: Some(destroy_callback),
    };

    // Store callbacks in the RwLock
    if let Ok(mut callbacks_guard) = QUEUE_CALLBACKS.write() {
        *callbacks_guard = Some(callbacks);
    } else {
        return Err(Error::new(bindings::hsa_status_t_HSA_STATUS_ERROR));
    }

    // Set callbacks
    let status = unsafe {
        bindings::rocprofiler_set_queue_callbacks(
            native_callbacks,
            ptr::null_mut(), // We're not using the data pointer
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        // If registration fails, remove our stored callbacks to maintain consistency
        if let Ok(mut callbacks_guard) = QUEUE_CALLBACKS.write() {
            *callbacks_guard = None;
        }
        return Err(Error::new(status));
    }

    Ok(())
}

/// Remove queue callbacks
pub fn remove_queue_callbacks() -> Result<()> {
    // Remove callbacks from ROCProfiler
    let status = unsafe { bindings::rocprofiler_remove_queue_callbacks() };

    // Clear stored callbacks
    if let Ok(mut callbacks_guard) = QUEUE_CALLBACKS.write() {
        *callbacks_guard = None;
    }

    Error::from_rocprofiler_error(status)
}

/// Start queue callbacks
pub fn start_queue_callbacks() -> Result<()> {
    let status = unsafe { bindings::rocprofiler_start_queue_callbacks() };
    Error::from_rocprofiler_error(status)
}

/// Stop queue callbacks
pub fn stop_queue_callbacks() -> Result<()> {
    let status = unsafe { bindings::rocprofiler_stop_queue_callbacks() };
    Error::from_rocprofiler_error(status)
}