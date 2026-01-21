// src/hip/mod.rs

// Private modules
pub mod device;
pub mod error;
pub mod event;
pub mod kernel;
pub mod memory;
pub mod module;
pub mod stream;
pub mod utils;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
#[allow(warnings)]
pub mod bindings;

// Public re-export of FFI for internal use
pub mod ffi;
pub mod memory_ext;

// Re-export the main components for the public API
pub use device::{Device, DeviceProperties, get_device_count, get_device_properties};
pub use error::{Error, Result};
pub use event::{Event, Timer, event_flags};
pub use kernel::{Function, stream_to_rocrand};
pub use memory::{DeviceMemory, MemoryInfo, PinnedMemory, memory_info};
pub use module::{Module, compile_and_load, load_module, load_module_data};
pub use stream::{Stream, stream_flags};
pub use utils::{
     Dim3, Version, calculate_grid_1d, calculate_grid_2d, calculate_grid_3d, is_hip_available, print_devices_info,
};

/// Get the number of devices
pub fn device_count() -> Result<i32> {
    device::get_device_count()
}

/// Initialize the HIP runtime
pub fn init() -> Result<()> {
    let error = unsafe { ffi::hipInit(0) };
    Error::from_hip_error(error)
}

/// Get the HIP driver version
pub fn driver_version() -> Result<i32> {
    let mut version = 0;
    let error = unsafe { ffi::hipDriverGetVersion(&mut version) };
    error::Error::from_hip_error_with_value(error, version)
}

/// Get the HIP runtime version
pub fn runtime_version() -> Result<i32> {
    let mut version = 0;
    let error = unsafe { ffi::hipRuntimeGetVersion(&mut version) };
    error::Error::from_hip_error_with_value(error, version)
}

/// Get the last error that occurred
pub fn get_last_error() -> Error {
    Error::new(unsafe { ffi::hipGetLastError() })
}

/// Synchronize the current device
pub fn device_synchronize() -> Result<()> {
    let error = unsafe { ffi::hipDeviceSynchronize() };
    Error::from_hip_error(error)
}

/// Reset the current device
pub fn device_reset() -> Result<()> {
    let error = unsafe { ffi::hipDeviceReset() };
    Error::from_hip_error(error)
}
