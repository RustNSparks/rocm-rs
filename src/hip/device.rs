// src/hip/device.rs

use crate::hip::error::{Error, Result};
use crate::hip::ffi;
use std::ffi::CStr;

/// Get the number of available devices
pub fn get_device_count() -> Result<i32> {
    let mut count = 0;
    let error = unsafe { ffi::hipGetDeviceCount(&mut count) };
    Error::from_hip_error_with_value(error, count)
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub total_const_mem: usize,
    pub major: i32,
    pub minor: i32,
    pub multi_processor_count: i32,
    pub l2_cache_size: i32,
    pub max_threads_per_multiprocessor: i32,
    pub compute_mode: i32,
    pub integrated: i32,
    pub can_map_host_memory: i32,
}

/// Get device properties for a given device
pub fn get_device_properties(device_id: i32) -> Result<DeviceProperties> {
    let mut props = unsafe { std::mem::zeroed::<ffi::hipDeviceProp_tR0600>() };
    let error = unsafe { ffi::hipGetDevicePropertiesR0600(&mut props, device_id) };

    if error != ffi::hipError_t_hipSuccess {
        return Err(Error::new(error));
    }

    let name = unsafe {
        let name_ptr = props.name.as_ptr() as *const i8;
        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
    };

    Ok(DeviceProperties {
        name,
        total_global_mem: props.totalGlobalMem,
        shared_mem_per_block: props.sharedMemPerBlock,
        regs_per_block: props.regsPerBlock,
        warp_size: props.warpSize,
        max_threads_per_block: props.maxThreadsPerBlock,
        max_threads_dim: props.maxThreadsDim,
        max_grid_size: props.maxGridSize,
        clock_rate: props.clockRate,
        memory_clock_rate: props.memoryClockRate,
        memory_bus_width: props.memoryBusWidth,
        total_const_mem: props.totalConstMem,
        major: props.major,
        minor: props.minor,
        multi_processor_count: props.multiProcessorCount,
        l2_cache_size: props.l2CacheSize,
        max_threads_per_multiprocessor: props.maxThreadsPerMultiProcessor,
        compute_mode: props.computeMode,
        integrated: props.integrated,
        can_map_host_memory: props.canMapHostMemory,
    })
}

/// A wrapper for HIP device operations
#[derive(Debug, Clone)]
pub struct Device {
    id: i32,
}

impl Device {
    /// Creates a new device with the given ID
    pub fn new(id: i32) -> Result<Self> {
        let count = get_device_count()?;
        if id < 0 || id >= count {
            return Err(Error::new(ffi::hipError_t_hipErrorInvalidDevice));
        }
        Ok(Self { id })
    }

    /// Get the current device
    pub fn current() -> Result<Self> {
        let mut device_id = 0;
        let error = unsafe { ffi::hipGetDevice(&mut device_id) };
        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }
        Ok(Self { id: device_id })
    }

    /// Get the device ID
    pub fn id(&self) -> i32 {
        self.id
    }

    /// Set this device as the current device
    pub fn set_current(&self) -> Result<()> {
        let error = unsafe { ffi::hipSetDevice(self.id) };
        Error::from_hip_error(error)
    }

    /// Synchronize this device
    pub fn synchronize(&self) -> Result<()> {
        // Save current device
        let current_device = Self::current()?;

        // Set this device as current
        self.set_current()?;

        // Synchronize
        let error = unsafe { ffi::hipDeviceSynchronize() };

        // Restore previous device
        current_device.set_current()?;

        Error::from_hip_error(error)
    }

    /// Reset this device
    pub fn reset(&self) -> Result<()> {
        // Save current device
        let current_device = Self::current()?;

        // Set this device as current
        self.set_current()?;

        // Reset
        let error = unsafe { ffi::hipDeviceReset() };

        // Restore previous device
        current_device.set_current()?;

        Error::from_hip_error(error)
    }

    /// Get the properties of this device
    pub fn properties(&self) -> Result<DeviceProperties> {
        get_device_properties(self.id)
    }
}
