// src/rocprofiler/profiler.rs

use crate::rocprofiler::error::{Error, Result};
use crate::rocprofiler::types::{Feature, ProfilerMode, InfoData, InfoKind};
use crate::rocprofiler::context::{Context, Properties};
use crate::rocprofiler::bindings;
use crate::hip;
use std::sync::Once;
use std::ffi::{CStr, CString};
use std::ptr;
use std::os::raw::c_void;

// Use a simpler initialization approach without static Mutex
static INIT: Once = Once::new();

/// Initialize the ROCProfiler system
///
/// This function initializes the ROCProfiler API. It's safe to call
/// multiple times, but will only perform the initialization once.
///
/// # Returns
///
/// * `Ok(())` - If initialization was successful
/// * `Err(Error)` - If initialization failed
pub fn init() -> Result<()> {
    // ROCProfiler currently doesn't require explicit initialization,
    // so this is just a placeholder for future functionality
    Ok(())
}

/// Get the ROCProfiler version as a string
///
/// # Returns
///
/// A string in the format "major.minor"
pub fn version_string() -> String {
    let major = unsafe { bindings::rocprofiler_version_major() };
    let minor = unsafe { bindings::rocprofiler_version_minor() };
    format!("{}.{}", major, minor)
}

// Callback used by iterate_info
unsafe extern "C" fn info_callback(
    info: bindings::rocprofiler_info_data_t,
    data: *mut c_void,
) -> u32 {
    if data.is_null() {
        return bindings::hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    let results = &mut *(data as *mut Vec<InfoData>);

    match InfoData::from_c(&info) {
        Ok(info_data) => {
            results.push(info_data);
            bindings::hsa_status_t_HSA_STATUS_SUCCESS
        },
        Err(_) => {
            bindings::hsa_status_t_HSA_STATUS_ERROR
        }
    }
}

/// Get the available metrics for a device
///
/// # Arguments
///
/// * `device` - Optional device to query metrics for. If None, uses the default device.
///
/// # Returns
///
/// * `Ok(Vec<InfoData>)` - List of available metrics
/// * `Err(Error)` - If an error occurred
pub fn get_metrics(device: Option<&hip::Device>) -> Result<Vec<InfoData>> {
    let mut results = Vec::new();

    // Create the HSA agent handle if device is provided
    let agent_ptr = match device {
        Some(dev) => {
            let agent = bindings::hsa_agent_t {
                handle: dev.id() as u64,
            };
            &agent as *const _
        },
        None => ptr::null(),
    };

    // Call the ROCProfiler API to get metrics
    let status = unsafe {
        bindings::rocprofiler_iterate_info(
            agent_ptr,
            InfoKind::Metric as u32,
            Some(info_callback),
            &mut results as *mut _ as *mut c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(results)
}

/// Get the available traces for a device
///
/// # Arguments
///
/// * `device` - Optional device to query traces for. If None, uses the default device.
///
/// # Returns
///
/// * `Ok(Vec<InfoData>)` - List of available traces
/// * `Err(Error)` - If an error occurred
pub fn get_traces(device: Option<&hip::Device>) -> Result<Vec<InfoData>> {
    let mut results = Vec::new();

    // Create the HSA agent handle if device is provided
    let agent_ptr = match device {
        Some(dev) => {
            let agent = bindings::hsa_agent_t {
                handle: dev.id() as u64,
            };
            &agent as *const _
        },
        None => ptr::null(),
    };

    // Call the ROCProfiler API to get traces
    let status = unsafe {
        bindings::rocprofiler_iterate_info(
            agent_ptr,
            InfoKind::Trace as u32,
            Some(info_callback),
            &mut results as *mut _ as *mut c_void,
        )
    };

    if status != bindings::hsa_status_t_HSA_STATUS_SUCCESS {
        return Err(Error::new(status));
    }

    Ok(results)
}

/// High-level profiler for ROCm applications
///
/// This struct provides a convenient way to profile GPU applications
/// using the ROCProfiler API.
pub struct Profiler {
    context: Context,
    device: hip::Device,
}

impl Profiler {
    /// Create a new profiler
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to profile
    /// * `features` - The metrics or traces to collect
    /// * `modes` - Profiler modes to use
    /// * `properties` - Optional properties for profiler configuration
    ///
    /// # Returns
    ///
    /// * `Ok(Profiler)` - The new profiler
    /// * `Err(Error)` - If profiler creation failed
    pub fn new(
        device: hip::Device,
        features: Vec<Feature>,
        modes: &[ProfilerMode],
        properties: Option<Properties>,
    ) -> Result<Self> {
        let context = Context::new(device.clone(), features, modes, properties)?;

        Ok(Self {
            context,
            device,
        })
    }

    /// Start profiling for a specific group
    ///
    /// # Arguments
    ///
    /// * `group_index` - The index of the group to start profiling
    pub fn start(&self, group_index: u32) -> Result<()> {
        self.context.start(group_index)
    }

    /// Stop profiling for a specific group
    ///
    /// # Arguments
    ///
    /// * `group_index` - The index of the group to stop profiling
    pub fn stop(&self, group_index: u32) -> Result<()> {
        self.context.stop(group_index)
    }

    /// Read profiling data for a specific group
    ///
    /// # Arguments
    ///
    /// * `group_index` - The index of the group to read data from
    pub fn read(&self, group_index: u32) -> Result<()> {
        self.context.read(group_index)
    }

    /// Get profiling data for a specific group
    ///
    /// # Arguments
    ///
    /// * `group_index` - The index of the group to get data from
    pub fn get_data(&mut self, group_index: u32) -> Result<()> {
        self.context.get_data(group_index)
    }

    /// Profile all groups (start, stop, read, get_data for each group)
    ///
    /// This is a convenience method that performs a complete profiling
    /// cycle for all groups.
    pub fn profile_all(&mut self) -> Result<()> {
        let group_count = self.context.group_count()?;

        for i in 0..group_count {
            self.start(i)?;
            self.stop(i)?;
            self.read(i)?;
            self.get_data(i)?;
        }

        Ok(())
    }

    /// Get all groups
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Group>)` - List of all groups
    /// * `Err(Error)` - If an error occurred
    pub fn get_groups(&self) -> Result<Vec<crate::rocprofiler::types::Group>> {
        let group_count = self.context.group_count()?;
        let mut groups = Vec::with_capacity(group_count as usize);

        for i in 0..group_count {
            groups.push(self.context.get_group(i)?);
        }

        Ok(groups)
    }

    /// Get the features
    pub fn features(&self) -> &[Feature] {
        self.context.features()
    }

    /// Get mutable access to the features
    pub fn features_mut(&mut self) -> &mut [Feature] {
        self.context.features_mut()
    }

    /// Reset a group
    ///
    /// # Arguments
    ///
    /// * `group_index` - The index of the group to reset
    pub fn reset(&self, group_index: u32) -> Result<()> {
        self.context.reset(group_index)
    }

    /// Get the device
    pub fn device(&self) -> &hip::Device {
        &self.device
    }

    /// Get the context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get mutable access to the context
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;
    use crate::rocprofiler::types::{Parameter, ParameterName};

    #[test]
    fn test_version_string() {
        let version = version_string();
        println!("ROCProfiler version: {}", version);
        // Just check that we get a version string that looks reasonable
        assert!(version.contains('.'));
    }

    #[test]
    fn test_get_metrics() {
        // This test will be skipped if no GPU is available
        if let Ok(device_count) = hip::device_count() {
            if device_count > 0 {
                if let Ok(device) = hip::Device::new(0) {
                    match get_metrics(Some(&device)) {
                        Ok(metrics) => {
                            println!("Found {} metrics", metrics.len());
                            for metric in &metrics {
                                if let InfoData::Metric(info) = metric {
                                    println!("  {}", info.name);
                                }
                            }
                        },
                        Err(e) => {
                            println!("Error getting metrics: {}", e);
                        }
                    }
                }
            }
        }
    }
}