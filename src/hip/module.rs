// src/hip/module.rs
//
// Module loading and management for HIP

use crate::hip::error::{Error, Result};
use crate::hip::ffi;
use crate::hip::kernel::Function;
use std::ffi::{CString, c_void};
use std::fs;
use std::path::Path;
use std::ptr;

/// A wrapper around a HIP module
pub struct Module {
    module: ffi::hipModule_t,
}

impl Module {
    /// Load a module from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let path_cstr = CString::new(path_str.as_bytes()).unwrap();

        let mut module = ptr::null_mut();
        let error = unsafe { ffi::hipModuleLoad(&mut module, path_cstr.as_ptr()) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { module })
    }

    /// Load a module from a string containing PTX code
    pub fn load_data(data: &str) -> Result<Self> {
        let data_cstr = CString::new(data).unwrap();

        let mut module = ptr::null_mut();
        let error =
            unsafe { ffi::hipModuleLoadData(&mut module, data_cstr.as_ptr() as *const c_void) };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { module })
    }

    /// Load a module from a file with options
    pub unsafe fn load_with_options<P: AsRef<Path>>(
        path: P,
        num_options: u32,
        options: *mut ffi::hipJitOption,
        option_values: *mut *mut c_void,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy();
        let path_cstr = CString::new(path_str.as_bytes()).unwrap();

        let mut module = ptr::null_mut();
        let error = unsafe {
            ffi::hipModuleLoadDataEx(
                &mut module,
                path_cstr.as_ptr() as *const c_void,
                num_options,
                options,
                option_values,
            )
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        Ok(Self { module })
    }

    /// Get a function from the module
    pub unsafe fn get_function(&self, name: &str) -> Result<Function> {
        Function::new(self.module, name)
    }

    /// Get a global variable from the module
    pub fn get_global<T>(&self, name: &str) -> Result<*mut T> {
        let name_cstr = CString::new(name).unwrap();

        let mut dev_ptr = ptr::null_mut();
        let mut size = 0usize;

        let error = unsafe {
            ffi::hipModuleGetGlobal(&mut dev_ptr, &mut size, self.module, name_cstr.as_ptr())
        };

        if error != ffi::hipError_t_hipSuccess {
            return Err(Error::new(error));
        }

        if size < std::mem::size_of::<T>() {
            return Err(Error::new(ffi::hipError_t_hipErrorInvalidValue));
        }

        Ok(dev_ptr as *mut T)
    }

    /// Get the raw module handle
    pub fn as_raw(&self) -> ffi::hipModule_t {
        self.module
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = ffi::hipModuleUnload(self.module);
                // We cannot handle errors in drop, so just ignore the result
            }
            self.module = ptr::null_mut();
        }
    }
}

/// Helper function to load a module from a file
pub fn load_module<P: AsRef<Path>>(path: P) -> Result<Module> {
    Module::load(path)
}

/// Helper function to load a module from data
pub fn load_module_data(data: &str) -> Result<Module> {
    Module::load_data(data)
}

/// Helper function to compile and load HIP code
pub fn compile_and_load(source: &str, options: &[String]) -> Result<Module> {
    // This is a placeholder for a function that would:
    // 1. Save the source to a temporary file
    // 2. Run hipcc to compile it
    // 3. Load the resulting binary
    //
    // A real implementation would depend on your build system
    // and how you want to handle compilation.
    //
    // For now, let's just show how it might work:
    use std::env::temp_dir;
    use std::process::Command;

    let temp_src_path = temp_dir().join("temp_kernel.cpp");
    let temp_bin_path = temp_dir().join("temp_kernel.hsaco");

    fs::write(&temp_src_path, source)
        .map_err(|_| Error::new(ffi::hipError_t_hipErrorInvalidValue))?;

    let mut cmd = Command::new("hipcc");
    cmd.arg("--genco");

    for opt in options {
        cmd.arg(opt);
    }

    cmd.arg("-o").arg(&temp_bin_path).arg(&temp_src_path);

    let status = cmd
        .status()
        .map_err(|_| Error::new(ffi::hipError_t_hipErrorInvalidValue))?;

    if !status.success() {
        return Err(Error::new(ffi::hipError_t_hipErrorInvalidValue));
    }

    Module::load(temp_bin_path)
}
