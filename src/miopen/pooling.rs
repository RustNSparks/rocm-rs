// src/miopen/pooling.rs

use std::ptr;
use std::os::raw::c_void;
use crate::miopen::ffi;
use crate::miopen::error::{Error, Result};
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;

/// Pooling mode
pub type PoolingMode = ffi::miopenPoolingMode_t;

/// Pooling workspace index mode
pub type PoolingWorkspaceIndexMode = ffi::miopenPoolingWorkspaceIndexMode_t;

/// Index type for pooling operations
pub type IndexType = ffi::miopenIndexType_t;

/// Safe wrapper for MIOpen pooling descriptor
pub struct PoolingDescriptor {
    desc: ffi::miopenPoolingDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for PoolingDescriptor {}
unsafe impl Sync for PoolingDescriptor {}

impl PoolingDescriptor {
    /// Create a new pooling descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreatePoolingDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the index data type for pooling layer
    pub fn set_index_type(&mut self, index_type: IndexType) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetPoolingIndexType(self.desc, index_type)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the index data type for pooling layer
    pub fn get_index_type(&self) -> Result<IndexType> {
        let mut index_type = 0;

        let status = unsafe {
            ffi::miopenGetPoolingIndexType(self.desc, &mut index_type)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(index_type)
    }

    /// Set the workspace index mode for pooling layer
    pub fn set_workspace_index_mode(&mut self, workspace_index: PoolingWorkspaceIndexMode) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetPoolingWorkSpaceIndexMode(self.desc, workspace_index)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the workspace index mode for pooling layer
    pub fn get_workspace_index_mode(&self) -> Result<PoolingWorkspaceIndexMode> {
        let mut workspace_index = 0;

        let status = unsafe {
            ffi::miopenGetPoolingWorkSpaceIndexMode(self.desc, &mut workspace_index)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(workspace_index)
    }

    /// Set a 2D pooling descriptor
    pub fn set_2d(&mut self, mode: PoolingMode, window_height: i32, window_width: i32,
                  pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32) -> Result<()> {
        let status = unsafe {
            ffi::miopenSet2dPoolingDescriptor(self.desc, mode, window_height, window_width,
                                              pad_h, pad_w, stride_h, stride_w)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get a 2D pooling descriptor
    pub fn get_2d(&self) -> Result<(PoolingMode, i32, i32, i32, i32, i32, i32)> {
        let mut mode = 0;
        let mut window_height = 0;
        let mut window_width = 0;
        let mut pad_h = 0;
        let mut pad_w = 0;
        let mut stride_h = 0;
        let mut stride_w = 0;

        let status = unsafe {
            ffi::miopenGet2dPoolingDescriptor(self.desc, &mut mode, &mut window_height, &mut window_width,
                                              &mut pad_h, &mut pad_w, &mut stride_h, &mut stride_w)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((mode, window_height, window_width, pad_h, pad_w, stride_h, stride_w))
    }

    /// Set an N-dimensional pooling descriptor
    pub fn set_nd(&mut self, mode: PoolingMode, window_dims: &[i32],
                  pads: &[i32], strides: &[i32]) -> Result<()> {
        let nb_dims = window_dims.len() as i32;

        if nb_dims as usize != pads.len() || nb_dims as usize != strides.len() {
            return Err(Error::new(ffi::miopenStatus_t_miopenStatusBadParm));
        }

        let status = unsafe {
            ffi::miopenSetNdPoolingDescriptor(self.desc, mode, nb_dims,
                                              window_dims.as_ptr(), pads.as_ptr(), strides.as_ptr())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get an N-dimensional pooling descriptor
    pub fn get_nd(&self, nb_dims_requested: i32) -> Result<(PoolingMode, i32, Vec<i32>, Vec<i32>, Vec<i32>)> {
        let mut mode = 0;
        let mut nb_dims = 0;
        let mut window_dims = vec![0; nb_dims_requested as usize];
        let mut pads = vec![0; nb_dims_requested as usize];
        let mut strides = vec![0; nb_dims_requested as usize];

        let status = unsafe {
            ffi::miopenGetNdPoolingDescriptor(self.desc, nb_dims_requested, &mut mode, &mut nb_dims,
                                              window_dims.as_mut_ptr(), pads.as_mut_ptr(), strides.as_mut_ptr())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((mode, nb_dims, window_dims, pads, strides))
    }

    /// Get the output dimensions of a pooling layer
    pub fn get_forward_output_dim(&self, tensor_desc: &TensorDescriptor) -> Result<(i32, i32, i32, i32)> {
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;

        let status = unsafe {
            ffi::miopenGetPoolingForwardOutputDim(self.desc, tensor_desc.as_raw(),
                                                  &mut n, &mut c, &mut h, &mut w)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((n, c, h, w))
    }

    /// Get the output dimensions of an N-dimensional pooling layer
    pub fn get_nd_forward_output_dim(&self, tensor_desc: &TensorDescriptor,
                                     dims_capacity: i32) -> Result<(i32, Vec<i32>)> {
        let mut tensor_dim_arr = vec![0; dims_capacity as usize];

        let status = unsafe {
            ffi::miopenGetPoolingNdForwardOutputDim(self.desc, tensor_desc.as_raw(),
                                                    dims_capacity, tensor_dim_arr.as_mut_ptr())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        // Return actual dimensions used
        let actual_dims = tensor_dim_arr.iter().position(|&x| x == 0).unwrap_or(tensor_dim_arr.len());
        tensor_dim_arr.truncate(actual_dims);

        Ok((actual_dims as i32, tensor_dim_arr))
    }

    /// Get the workspace size required for pooling operations
    pub fn get_workspace_size(&self, y_desc: &TensorDescriptor) -> Result<usize> {
        let mut workspace_size = 0;

        let status = unsafe {
            ffi::miopenPoolingGetWorkSpaceSizeV2(self.desc, y_desc.as_raw(), &mut workspace_size)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(workspace_size)
    }

    /// Execute a forward pooling operation
    pub fn forward(
        &self,
        handle: &Handle,
        alpha: &[u8],
        x_desc: &TensorDescriptor,
        x: *const c_void,
        beta: &[u8],
        y_desc: &TensorDescriptor,
        y: *mut c_void,
        do_backward: bool,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenPoolingForward(
                handle.as_raw(),
                self.desc,
                alpha.as_ptr() as *const c_void,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const c_void,
                y_desc.as_raw(),
                y,
                do_backward,
                workspace,
                workspace_size,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute a backward pooling operation
    pub fn backward(
        &self,
        handle: &Handle,
        alpha: &[u8],
        y_desc: &TensorDescriptor,
        y: *const c_void,
        dy_desc: &TensorDescriptor,
        dy: *const c_void,
        x_desc: &TensorDescriptor,
        x: *const c_void,
        beta: &[u8],
        dx_desc: &TensorDescriptor,
        dx: *mut c_void,
        workspace: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenPoolingBackward(
                handle.as_raw(),
                self.desc,
                alpha.as_ptr() as *const c_void,
                y_desc.as_raw(),
                y,
                dy_desc.as_raw(),
                dy,
                x_desc.as_raw(),
                x,
                beta.as_ptr() as *const c_void,
                dx_desc.as_raw(),
                dx,
                workspace,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenPoolingDescriptor_t {
        self.desc
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyPoolingDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}