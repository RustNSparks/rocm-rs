// src/hip/mod.rs

// Private modules
mod error;
mod device;
mod memory;
mod stream;
mod event;
mod utils;
pub mod kernel;
pub mod module;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
pub(crate) mod bindings;

// Public re-export of FFI for internal use
pub mod ffi;

// Re-export the main components for the public API
pub use error::{Error, Result};
pub use device::{Device, DeviceProperties, get_device_count, get_device_properties};
pub use memory::{DeviceMemory, PinnedMemory, MemoryInfo, memory_info};
pub use stream::{Stream, stream_flags};
pub use event::{Event, event_flags, Timer};
pub use utils::{DeviceGuard, Version, Dim3,
                print_devices_info, run_on_device,
                calculate_grid_1d, calculate_grid_2d, calculate_grid_3d,
                copy_kind, host_mem_flags, is_hip_available};
pub use kernel::{Function, KernelArg, stream_to_rocrand, launch_kernel};
pub use module::{Module, load_module, load_module_data, compile_and_load};

// Re-export macros
pub use crate::{launch_kernel, kernel_launcher};

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