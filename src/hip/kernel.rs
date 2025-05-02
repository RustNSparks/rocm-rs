// src/hip/kernel.rs
//
// Kernel launching functions for HIP

use std::ffi::{c_void, CString};
use std::ptr;
use crate::hip::error::{Error, Result};
use crate::hip::ffi;
use crate::hip::utils::Dim3;
use crate::hip::Stream;

/// A wrapper around a HIP function (kernel)
pub struct Function {
    function: ffi::hipFunction_t,
}

impl Function {
    /// Create a new function from a module and function name
    pub fn new(module: ffi::hipModule_t, name: &str) -> Result<Self> {
        let func_name = CString::new(name).unwrap();
        let mut function = ptr::null_mut();
        
        let error = unsafe {
            ffi::hipModuleGetFunction(&mut function, module, func_name.as_ptr())
        };
        
        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }
        
        Ok(Self { function })
    }
    
    /// Launch the kernel with the given parameters
    pub fn launch(
        &self,
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem_bytes: u32,
        stream: Option<&Stream>,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        let stream_ptr = match stream {
            Some(s) => s.as_raw(),
            None => ptr::null_mut(),
        };
        
        let error = unsafe {
            ffi::hipModuleLaunchKernel(
                self.function,
                grid_dim.x, grid_dim.y, grid_dim.z,
                block_dim.x, block_dim.y, block_dim.z,
                shared_mem_bytes,
                stream_ptr,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(), // extra
            )
        };
        
        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }
        
        Ok(())
    }
    
    /// Get the raw function handle
    pub fn as_raw(&self) -> ffi::hipFunction_t {
        self.function
    }
}

/// A trait for types that can be passed as kernel arguments
pub trait KernelArg {
    /// Get a pointer to the argument value
    fn as_ptr(&self) -> *const c_void;
}

// Implement KernelArg for common types
macro_rules! impl_kernel_arg {
    ($($t:ty),*) => {
        $(
            impl KernelArg for $t {
                fn as_ptr(&self) -> *const c_void {
                    self as *const $t as *const c_void
                }
            }
        )*
    };
}

impl_kernel_arg!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// Helper for safe kernel launches via the hipLaunchKernel API
/// Launch a HIP kernel using the driver API
#[macro_export]
macro_rules! launch_kernel {
    ($func:expr, $grid:expr, $block:expr, $shared_mem:expr, $stream:expr, $($arg:expr),*) => {
        {
            let mut args = vec![];
            $(
                args.push($arg.as_ptr() as *mut std::ffi::c_void);
            )*
            
            $func.launch($grid, $block, $shared_mem, $stream, &mut args)
        }
    };
}

/// Launch a kernel via the runtime API
/// 
/// # Safety
/// 
/// This function is unsafe because it takes a raw function pointer and
/// arguments that must match the function signature.
pub unsafe fn launch_kernel(
    kernel_func_ptr: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    shared_mem_bytes: u32,
    stream: Option<&Stream>,
    args: &[*mut c_void],
) -> Result<()>
{
    let stream_ptr = match stream {
        Some(s) => s.as_raw(),
        None => ptr::null_mut(),
    };
    
    let native_grid_dim = grid_dim.to_native();
    let native_block_dim = block_dim.to_native();
    
    let error = unsafe {
        ffi::hipLaunchKernel(
            kernel_func_ptr,
            native_grid_dim,
            native_block_dim,
            args.as_ptr() as *mut *mut c_void,
            shared_mem_bytes.try_into().unwrap(),
            stream_ptr,
        )
    };
    
    if error != ffi::hipError_t_hipSuccess {
        return Err(Error::new(error));
    }
    
    Ok(())
}

/// Macro to generate a kernel launcher function
/// 
/// This macro generates a function that takes the grid dimensions, block dimensions,
/// shared memory size, stream, and kernel arguments, and launches the kernel.
#[macro_export]
macro_rules! kernel_launcher {
    ($name:ident, $func:path, $($arg_ty:ty),*) => {
        pub fn $name(
            grid_dim: $crate::hip::utils::Dim3,
            block_dim: $crate::hip::utils::Dim3,
            shared_mem_bytes: u32,
            stream: Option<&$crate::hip::Stream>,
            $($arg:$arg_ty),*
        ) -> $crate::hip::error::Result<()> {
            unsafe {
                let args: Vec<*mut std::ffi::c_void> = vec![
                    $(&$arg as *const $arg_ty as *mut std::ffi::c_void),*
                ];
                
                $crate::hip::kernel::launch_kernel(
                    $func,
                    grid_dim,
                    block_dim,
                    shared_mem_bytes,
                    stream,
                    &args,
                )
            }
        }
    };
}

/// Helper function to convert a Stream reference to the rocrand stream type
pub fn stream_to_rocrand(stream: &Stream) -> crate::rocrand::bindings::hipStream_t {
    // Safe cast because both represent the same underlying HIP stream
    stream.as_raw() as crate::rocrand::bindings::hipStream_t
}