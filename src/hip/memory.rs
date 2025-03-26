// src/hip/memory.rs

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use crate::hip::ffi;
use crate::hip::error::{Error, Result};

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
    count: usize,
    phantom: PhantomData<T>,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl<T: Send> Send for DeviceMemory<T> {}
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

impl<T> DeviceMemory<T> {
    /// Allocate device memory for a number of elements
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
        let error = unsafe { ffi::hipMalloc(&mut ptr, size) };

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
        self.count
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

        let error = unsafe {
            ffi::hipMemset(self.ptr, value, self.size)
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Asynchronously copy data from host to device
    pub fn copy_from_host_async(&mut self, data: &[T], stream: &crate::hip::Stream) -> Result<()> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }

        let copy_size = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let error = unsafe {
            ffi::hipMemcpyAsync(
                self.ptr,
                data.as_ptr() as *const c_void,
                copy_size,
                ffi::hipMemcpyKind_hipMemcpyHostToDevice,
                stream.as_raw(),
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
    }

    /// Asynchronously copy data from device to host
    pub fn copy_to_host_async(&self, data: &mut [T], stream: &crate::hip::Stream) -> Result<()> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }

        let copy_size = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let error = unsafe {
            ffi::hipMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                copy_size,
                ffi::hipMemcpyKind_hipMemcpyDeviceToHost,
                stream.as_raw(),
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(())
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

        unsafe {
            std::slice::from_raw_parts(self.ptr as *const T, self.count)
        }
    }

    /// Get the host pointer as a mutable slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        if self.ptr.is_null() || self.count == 0 {
            return &mut [];
        }

        unsafe {
            std::slice::from_raw_parts_mut(self.ptr as *mut T, self.count)
        }
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
        let error = unsafe {
            ffi::hipHostGetDevicePointer(&mut device_ptr, self.ptr, 0)
        };

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