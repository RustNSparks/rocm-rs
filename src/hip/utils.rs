// src/hip/utils.rs

use crate::hip::error::Result;
use crate::hip::Device;

/// A simple RAII (Resource Acquisition Is Initialization) guard 
/// to set a device as current and restore the previous device when dropped
pub struct DeviceGuard {
    previous_device: i32,
}

impl DeviceGuard {
    /// Create a new device guard, setting the given device as current
    pub fn new(device_id: i32) -> Result<Self> {
        // Get the current device
        let device = Device::current()?;
        let previous_device = device.id();

        // Set the new device as current
        let new_device = Device::new(device_id)?;
        new_device.set_current()?;

        Ok(Self { previous_device })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        // Restore the previous device
        if let Ok(device) = Device::new(self.previous_device) {
            let _ = device.set_current();
            // We cannot handle errors in drop, so just ignore the result
        }
    }
}

/// Get a description of all devices in the system
pub fn print_devices_info() -> Result<String> {
    let count = crate::hip::get_device_count()?;
    let mut output = String::new();

    output.push_str(&format!("Found {} HIP device(s)\n", count));

    for i in 0..count {
        let device = Device::new(i)?;
        let props = device.properties()?;

        output.push_str(&format!("\nDevice {}: {}\n", i, props.name));
        output.push_str(&format!("  Compute capability: {}.{}\n", props.major, props.minor));
        output.push_str(&format!("  Total memory: {} MB\n", props.total_global_mem / (1024 * 1024)));
        output.push_str(&format!("  Clock rate: {} MHz\n", props.clock_rate / 1000));
        output.push_str(&format!("  Multi-processor count: {}\n", props.multi_processor_count));
        output.push_str(&format!("  Max threads per block: {}\n", props.max_threads_per_block));
        output.push_str(&format!("  Max threads per multiprocessor: {}\n", props.max_threads_per_multiprocessor));
        output.push_str(&format!("  Warp size: {}\n", props.warp_size));
        output.push_str(&format!("  Max dimensions of a grid: [{}, {}, {}]\n",
                                 props.max_grid_size[0],
                                 props.max_grid_size[1],
                                 props.max_grid_size[2]));
        output.push_str(&format!("  Max dimensions of a block: [{}, {}, {}]\n",
                                 props.max_threads_dim[0],
                                 props.max_threads_dim[1],
                                 props.max_threads_dim[2]));
        output.push_str(&format!("  Shared memory per block: {} KB\n", props.shared_mem_per_block / 1024));
        output.push_str(&format!("  Registers per block: {}\n", props.regs_per_block));
        output.push_str(&format!("  L2 cache size: {} KB\n", props.l2_cache_size / 1024));
        output.push_str(&format!("  Memory clock rate: {} MHz\n", props.memory_clock_rate / 1000));
        output.push_str(&format!("  Memory bus width: {} bits\n", props.memory_bus_width));
        output.push_str(&format!("  Integrated: {}\n", props.integrated));
        output.push_str(&format!("  Can map host memory: {}\n", props.can_map_host_memory));
    }

    Ok(output)
}

/// Convenient constants for memory copy types
pub mod copy_kind {
    use crate::hip::ffi;

    /// Host to Host memory copy
    pub const HOST_TO_HOST: u32 = ffi::hipMemcpyKind_hipMemcpyHostToHost;

    /// Host to Device memory copy
    pub const HOST_TO_DEVICE: u32 = ffi::hipMemcpyKind_hipMemcpyHostToDevice;

    /// Device to Host memory copy
    pub const DEVICE_TO_HOST: u32 = ffi::hipMemcpyKind_hipMemcpyDeviceToHost;

    /// Device to Device memory copy
    pub const DEVICE_TO_DEVICE: u32 = ffi::hipMemcpyKind_hipMemcpyDeviceToDevice;

    /// Automatic detection of memory copy type
    pub const DEFAULT: u32 = ffi::hipMemcpyKind_hipMemcpyDefault;
}

/// Convenient constants for host memory flags
pub mod host_mem_flags {
    use crate::hip::ffi;

    /// Default host memory allocation flag
    pub const DEFAULT: u32 = ffi::hipHostMallocDefault;

    /// Memory is accessible from all GPUs
    pub const PORTABLE: u32 = ffi::hipHostMallocPortable;

    /// Map the allocation into the GPU address space
    pub const MAPPED: u32 = ffi::hipHostMallocMapped;

    /// Allocate the memory as write-combined
    pub const WRITE_COMBINED: u32 = ffi::hipHostMallocWriteCombined;

    /// Memory will be allocated in a NUMA-optimized way
    pub const NUMA_USER: u32 = ffi::hipHostMallocNumaUser;

    /// Memory coherent with the GPU
    pub const COHERENT: u32 = ffi::hipHostMallocCoherent;

    /// Memory not coherent with the GPU
    pub const NON_COHERENT: u32 = ffi::hipHostMallocNonCoherent;
}

/// Wrapper for HIP version information
pub struct Version {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

impl Version {
    /// Get the HIP driver version
    pub fn driver() -> Result<Self> {
        let version = crate::hip::driver_version()?;

        // HIP versions are encoded as (10000*major + 100*minor + patch)
        let major = version / 10000;
        let minor = (version % 10000) / 100;
        let patch = version % 100;

        Ok(Self { major, minor, patch })
    }

    /// Get the HIP runtime version
    pub fn runtime() -> Result<Self> {
        let version = crate::hip::runtime_version()?;

        // HIP versions are encoded as (10000*major + 100*minor + patch)
        let major = version / 10000;
        let minor = (version % 10000) / 100;
        let patch = version % 100;

        Ok(Self { major, minor, patch })
    }

    /// Convert to a string representation
    pub fn to_string(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Wrapper function to run a function on a specific device
pub fn run_on_device<F, T>(device_id: i32, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let _guard = DeviceGuard::new(device_id)?;
    f()
}

/// Convenient struct for 3D dimensions
#[derive(Debug, Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Create a new 1D dimension
    pub fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new 2D dimension
    pub fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new 3D dimension
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Convert to the native HIP dim3 structure
    pub fn to_native(&self) -> crate::hip::ffi::dim3 {
        crate::hip::ffi::dim3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self::new_1d(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self::new_2d(x, y)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new_3d(x, y, z)
    }
}

/// Calculate optimal grid dimensions for a 1D problem
pub fn calculate_grid_1d(total_elements: u32, block_size: u32) -> Dim3 {
    let grid_size = (total_elements + block_size - 1) / block_size;
    Dim3::new_1d(grid_size)
}

/// Calculate optimal grid dimensions for a 2D problem
pub fn calculate_grid_2d(width: u32, height: u32, block_x: u32, block_y: u32) -> Dim3 {
    let grid_x = (width + block_x - 1) / block_x;
    let grid_y = (height + block_y - 1) / block_y;
    Dim3::new_2d(grid_x, grid_y)
}

/// Calculate optimal grid dimensions for a 3D problem
pub fn calculate_grid_3d(width: u32, height: u32, depth: u32, block_x: u32, block_y: u32, block_z: u32) -> Dim3 {
    let grid_x = (width + block_x - 1) / block_x;
    let grid_y = (height + block_y - 1) / block_y;
    let grid_z = (depth + block_z - 1) / block_z;
    Dim3::new_3d(grid_x, grid_y, grid_z)
}

/// Helper function to get the next multiple of a value
pub fn next_multiple(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}

/// Helper function to determine if HIP is available
pub fn is_hip_available() -> bool {
    match crate::hip::device_count() {
        Ok(count) => count > 0,
        Err(_) => false,
    }
}