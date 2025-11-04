// src/hip/memory.rs

#[cfg(feature = "macros")]
pub mod memory_ext;

use crate::hip::error::{Error, Result};
use crate::hip::kernel::AsKernelArg;
use crate::hip::{Stream, ffi};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::{mem, ptr};

pub type KernelArg = *mut c_void;

/// Information about available and used memory on the device
#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    pub free: usize,
    pub total: usize,
}

/// Get memory information for the current device
pub fn memory_info() -> Result<MemoryInfo> {
    let mut free = 0;
    let mut total = 0;
    let error = unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };
    if error != ffi::hipError_t_hipSuccess {
        return Err(Error::new(error));
    }
    Ok(MemoryInfo { free, total })
}

/// Safe wrapper for hip device memory
pub struct DeviceMemory<T> {
    ptr: *mut c_void,
    size: usize,
    phantom: PhantomData<T>,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl<T: Send> Send for DeviceMemory<T> {}
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

#[derive(Clone)]
pub struct PendingCopy<T> {
    inner: Vec<T>,
}

impl<T> PendingCopy<T> {
    pub fn synchronize(self) -> Vec<T> {
        self.inner
    }
}

pub trait SynchronizeCopies {
    type Output;
    /// unwraps PendingCopy<T>, use only if synchronize was called manually before
    unsafe fn finalize(self) -> Self::Output;
}

impl<T> SynchronizeCopies for PendingCopy<T> {
    type Output = Vec<T>;

    unsafe fn finalize(self) -> Self::Output {
        self.synchronize()
    }
}

impl<T, Rest> SynchronizeCopies for (PendingCopy<T>, Rest)
where
    Rest: SynchronizeCopies,
{
    type Output = (Vec<T>, Rest::Output);

    unsafe fn finalize(self) -> Self::Output {
        let (pending, rest) = self;
        let vec = pending.synchronize();
        let rest_out = unsafe { rest.finalize() };
        (vec, rest_out)
    }
}

impl SynchronizeCopies for () {
    type Output = ();

    unsafe fn finalize(self) -> Self::Output {
        ()
    }
}

impl<T> DeviceMemory<T> {
    /// Allocate device memory for a number of elements
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size: 0,
                phantom: PhantomData,
            });
        }

        let size = count * size_of::<T>();
        let mut ptr = ptr::null_mut();
        let error = unsafe { ffi::hipMalloc(&mut ptr, size) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self {
            ptr,
            size,
            phantom: PhantomData,
        })
    }

    /// Get the device pointer
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the number of elements
    pub fn count(&self) -> usize {
        self.size / size_of::<T>()
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }

        let copy_size = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let error = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                copy_size,
                ffi::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }

        let copy_size = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let error = unsafe {
            ffi::hipMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                copy_size,
                ffi::hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Copy data from another device memory
    pub fn copy_from_device(&mut self, src: &DeviceMemory<T>) -> Result<()> {
        if self.ptr.is_null() || src.ptr.is_null() {
            return Ok(());
        }

        let copy_size = std::cmp::min(self.size, src.size);
        let error = unsafe {
            ffi::hipMemcpy(
                self.ptr,
                src.ptr,
                copy_size,
                ffi::hipMemcpyKind_hipMemcpyDeviceToDevice,
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Set memory to a value
    pub fn memset(&mut self, value: i32) -> Result<()> {
        if self.ptr.is_null() {
            return Ok(());
        }

        let error = unsafe { ffi::hipMemset(self.ptr, value, self.size) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    pub fn copy_from_host_async<I: Into<Vec<T>>>(&self, source: I, stream: &Stream) -> Result<()> {
        let source = Into::<Vec<T>>::into(source);

        // Check for empty source or potentially uninitialized buffer early
        if source.is_empty() {
            return Ok(());
        }
        // Check if self.ptr is null if your struct allows for uninitialized state
        // if self.ptr.is_null() { return Err(/* Appropriate error */); }

        let required_bytes = source.len().saturating_mul(mem::size_of::<T>()); // Use saturating_mul just in case

        // Check if the source data fits within the allocated buffer size
        if required_bytes > self.size {
            return Err(Error::new(ffi::hipError_t_hipErrorInvalidValue));
        }

        // Only proceed with copy if there are bytes to copy (handles ZSTs correctly)
        if required_bytes == 0 {
            return Ok(());
        }

        let error = unsafe {
            ffi::hipMemcpyAsync(
                self.ptr, // Assuming self.ptr is *mut c_void or compatible
                source.as_ptr() as *const c_void,
                required_bytes, // Copy the exact size needed for the source slice
                ffi::hipMemcpyKind_hipMemcpyHostToDevice,
                stream.as_raw(),
            )
        };

        // Check hipMemcpyAsync result
        if error != ffi::hipError_t_hipSuccess {
            Err(Error::new(error)) // Assumes Error::new handles hipError_t
        } else {
            Ok(())
        }
    }

    /// Asynchronously copies data from this device buffer to a host slice `dest`.
    ///
    /// Copies `dest.len() * size_of::<T>()` bytes.
    ///
    /// # Arguments
    /// * `dest` - The host slice to copy data into.
    /// * `stream` - The HIP stream to perform the copy operation on.
    ///
    /// # Errors
    /// - Returns `Error::CopySizeMismatch` if the destination slice (`dest.len() * size_of::<T>()`)
    ///   requests more bytes than are available in this GPU buffer (`self.size`).
    /// - Returns other `hip::Error` variants if the `hipMemcpyAsync` call fails.
    ///
    /// # Notes
    /// - This operation is asynchronous. The caller must synchronize the `stream`
    ///   (e.g., via `stream.synchronize()`) before accessing the data in the `dest`
    ///   slice on the host.
    /// - If `dest` is empty, the function returns `Ok(())` immediately.
    pub fn copy_to_host_async<'a>(
        &self,
        mut dest: Vec<T>,
        stream: &Stream,
    ) -> Result<PendingCopy<T>> {
        // Check for empty destination or potentially uninitialized buffer early
        if dest.is_empty() {
            return Ok(PendingCopy { inner: dest });
        }
        // Check if self.ptr is null if your struct allows for uninitialized state
        // if self.ptr.is_null() { return Err(/* Appropriate error */); }

        let required_bytes = dest.len().saturating_mul(mem::size_of::<T>());

        // Check if the GPU buffer has enough data to fill the destination slice
        if required_bytes > self.size {
            return Err(Error::new(ffi::hipError_t_hipErrorOutOfMemory));
        }

        // Only proceed with copy if there are bytes to copy (handles ZSTs correctly)
        if required_bytes == 0 {
            return Ok(PendingCopy { inner: dest });
        }

        let error = unsafe {
            ffi::hipMemcpyAsync(
                dest.as_mut_ptr() as *mut c_void,
                self.ptr,       // Assuming self.ptr is *const c_void or compatible
                required_bytes, // Copy the exact size requested by the dest slice
                ffi::hipMemcpyKind_hipMemcpyDeviceToHost,
                stream.as_raw(),
            )
        };

        // Check hipMemcpyAsync result
        if error != ffi::hipError_t_hipSuccess {
            Err(Error::new(error)) // Assumes Error::new handles hipError_t
        } else {
            Ok(PendingCopy { inner: dest })
        }
    }
}

impl<T> AsKernelArg for DeviceMemory<T> {
    fn as_kernel_arg(&self) -> KernelArg {
        &(self.ptr) as *const _ as KernelArg
    }
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = ffi::hipFree(self.ptr);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Safe wrapper for pinned (page-locked) host memory
pub struct PinnedMemory<T> {
    ptr: *mut c_void,
    size: usize,
    count: usize,
    phantom: PhantomData<T>,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl<T: Send> Send for PinnedMemory<T> {}
unsafe impl<T: Sync> Sync for PinnedMemory<T> {}

impl<T> PinnedMemory<T> {
    /// Allocate pinned host memory for a number of elements
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size: 0,
                count: 0,
                phantom: PhantomData,
            });
        }

        let size = count * std::mem::size_of::<T>();
        let mut ptr = ptr::null_mut();
        let error = unsafe { ffi::hipHostMalloc(&mut ptr, size, 0) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self {
            ptr,
            size,
            count,
            phantom: PhantomData,
        })
    }

    /// Allocate pinned host memory with specific flags
    pub fn with_flags(count: usize, flags: u32) -> Result<Self> {
        if count == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size: 0,
                count: 0,
                phantom: PhantomData,
            });
        }

        let size = count * std::mem::size_of::<T>();
        let mut ptr = ptr::null_mut();
        let error = unsafe { ffi::hipHostMalloc(&mut ptr, size, flags) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self {
            ptr,
            size,
            count,
            phantom: PhantomData,
        })
    }

    /// Get the host pointer as a slice
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() || self.count == 0 {
            return &[];
        }

        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.count) }
    }

    /// Get the host pointer as a mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.ptr.is_null() || self.count == 0 {
            return &mut [];
        }

        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.count) }
    }

    /// Get the raw host pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get the raw mutable host pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the number of elements
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the device pointer for this pinned memory
    pub fn get_device_pointer(&self) -> Result<*mut c_void> {
        if self.ptr.is_null() {
            return Ok(ptr::null_mut());
        }

        let mut device_ptr = ptr::null_mut();
        let error = unsafe { ffi::hipHostGetDevicePointer(&mut device_ptr, self.ptr, 0) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(device_ptr)
    }
}

impl<T> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = ffi::hipHostFree(self.ptr);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.ptr = ptr::null_mut();
        }
    }
}
