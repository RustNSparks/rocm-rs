// src/miopen/convolution.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::c_void;
use std::ptr;

/// Convolution mode
pub type ConvolutionMode = ffi::miopenConvolutionMode_t;

/// Convolution forward algorithm
pub type ConvFwdAlgorithm = ffi::miopenConvFwdAlgorithm_t;

/// Convolution backward data algorithm
pub type ConvBwdDataAlgorithm = ffi::miopenConvBwdDataAlgorithm_t;

/// Convolution backward weights algorithm
pub type ConvBwdWeightsAlgorithm = ffi::miopenConvBwdWeightsAlgorithm_t;

/// General convolution algorithm
pub type ConvAlgorithm = ffi::miopenConvAlgorithm_t;

/// Padding mode
pub type PaddingMode = ffi::miopenPaddingMode_t;

/// Convolution attribute
pub type ConvolutionAttribute = ffi::miopenConvolutionAttrib_t;

/// Performance result for convolution algorithms
pub type ConvolutionPerf = ffi::miopenConvAlgoPerf_t;

/// Safe wrapper for MIOpen convolution descriptor
pub struct ConvolutionDescriptor {
    desc: ffi::miopenConvolutionDescriptor_t,
}

impl ConvolutionDescriptor {
    /// Create a new convolution descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateConvolutionDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Initialize a 2D convolution descriptor
    pub fn init_2d(
        &mut self,
        c_mode: ConvolutionMode,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenInitConvolutionDescriptor(
                self.desc, c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Initialize an N-dimensional convolution descriptor
    pub fn init_nd(
        &mut self,
        padA: &[i32],
        strideA: &[i32],
        dilationA: &[i32],
        c_mode: ConvolutionMode,
    ) -> Result<()> {
        let spatial_dim = padA.len() as i32;

        if spatial_dim as usize != strideA.len() || spatial_dim as usize != dilationA.len() {
            return Err(Error::new(ffi::miopenStatus_t_miopenStatusBadParm));
        }

        let status = unsafe {
            ffi::miopenInitConvolutionNdDescriptor(
                self.desc,
                spatial_dim,
                padA.as_ptr(),
                strideA.as_ptr(),
                dilationA.as_ptr(),
                c_mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the spatial dimension of the convolution descriptor
    pub fn get_spatial_dim(&self) -> Result<i32> {
        let mut spatial_dim = 0;

        let status = unsafe { ffi::miopenGetConvolutionSpatialDim(self.desc, &mut spatial_dim) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(spatial_dim)
    }

    /// Get the details of a 2D convolution descriptor
    pub fn get_2d(&self) -> Result<(ConvolutionMode, i32, i32, i32, i32, i32, i32)> {
        let mut c_mode = 0;
        let mut pad_h = 0;
        let mut pad_w = 0;
        let mut stride_h = 0;
        let mut stride_w = 0;
        let mut dilation_h = 0;
        let mut dilation_w = 0;

        let status = unsafe {
            ffi::miopenGetConvolutionDescriptor(
                self.desc,
                &mut c_mode,
                &mut pad_h,
                &mut pad_w,
                &mut stride_h,
                &mut stride_w,
                &mut dilation_h,
                &mut dilation_w,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((
            c_mode, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        ))
    }

    /// Get the details of an N-dimensional convolution descriptor
    pub fn get_nd(
        &self,
        requested_spatial_dim: i32,
    ) -> Result<(i32, Vec<i32>, Vec<i32>, Vec<i32>, ConvolutionMode)> {
        let mut spatial_dim = 0;
        let mut padA = vec![0; requested_spatial_dim as usize];
        let mut strideA = vec![0; requested_spatial_dim as usize];
        let mut dilationA = vec![0; requested_spatial_dim as usize];
        let mut c_mode = 0;

        let status = unsafe {
            ffi::miopenGetConvolutionNdDescriptor(
                self.desc,
                requested_spatial_dim,
                &mut spatial_dim,
                padA.as_mut_ptr(),
                strideA.as_mut_ptr(),
                dilationA.as_mut_ptr(),
                &mut c_mode,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((spatial_dim, padA, strideA, dilationA, c_mode))
    }

    /// Get the number of groups for group/depthwise convolution
    pub fn get_group_count(&self) -> Result<i32> {
        let mut group_count = 0;

        let status = unsafe { ffi::miopenGetConvolutionGroupCount(self.desc, &mut group_count) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(group_count)
    }

    /// Set the number of groups for group/depthwise convolution
    pub fn set_group_count(&mut self, group_count: i32) -> Result<()> {
        let status = unsafe { ffi::miopenSetConvolutionGroupCount(self.desc, group_count) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set output padding for 2D transpose convolution
    pub fn set_transpose_conv_output_padding(&mut self, adj_h: i32, adj_w: i32) -> Result<()> {
        let status = unsafe { ffi::miopenSetTransposeConvOutputPadding(self.desc, adj_h, adj_w) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set output padding for N-dimensional transpose convolution
    pub fn set_transpose_conv_nd_output_padding(&mut self, adjA: &[i32]) -> Result<()> {
        let spatial_dim = adjA.len() as i32;

        let status = unsafe {
            ffi::miopenSetTransposeConvNdOutputPadding(self.desc, spatial_dim, adjA.as_ptr())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the output dimensions of a 2D convolution
    pub fn get_forward_output_dim(
        &self,
        input_desc: &TensorDescriptor,
        filter_desc: &TensorDescriptor,
    ) -> Result<(i32, i32, i32, i32)> {
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;

        let status = unsafe {
            ffi::miopenGetConvolutionForwardOutputDim(
                self.desc,
                input_desc.as_raw(),
                filter_desc.as_raw(),
                &mut n,
                &mut c,
                &mut h,
                &mut w,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((n, c, h, w))
    }

    /// Get the output dimensions of an N-dimensional convolution
    pub fn get_nd_forward_output_dim(
        &self,
        input_desc: &TensorDescriptor,
        filter_desc: &TensorDescriptor,
        dims_capacity: usize,
    ) -> Result<(i32, Vec<i32>)> {
        let mut n_dim = 0;
        let mut output_dims = vec![0; dims_capacity];

        let status = unsafe {
            ffi::miopenGetConvolutionNdForwardOutputDim(
                self.desc,
                input_desc.as_raw(),
                filter_desc.as_raw(),
                &mut n_dim,
                output_dims.as_mut_ptr(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((n_dim, output_dims))
    }

    /// Set an attribute for the convolution descriptor
    pub fn set_attribute(&mut self, attr: ConvolutionAttribute, value: i32) -> Result<()> {
        let status = unsafe { ffi::miopenSetConvolutionAttribute(self.desc, attr, value) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get an attribute from the convolution descriptor
    pub fn get_attribute(&self, attr: ConvolutionAttribute) -> Result<i32> {
        let mut value = 0;

        let status = unsafe { ffi::miopenGetConvolutionAttribute(self.desc, attr, &mut value) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(value)
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyConvolutionDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

/// Query the workspace size required for a forward convolution algorithm
pub fn get_convolution_forward_workspace_size(
    handle: &Handle,
    w_desc: &TensorDescriptor,
    x_desc: &TensorDescriptor,
    conv_desc: &ConvolutionDescriptor,
    y_desc: &TensorDescriptor,
) -> Result<usize> {
    let mut workspace_size = 0;

    let status = unsafe {
        ffi::miopenConvolutionForwardGetWorkSpaceSize(
            handle.as_raw(),
            w_desc.as_raw(),
            x_desc.as_raw(),
            conv_desc.as_raw(),
            y_desc.as_raw(),
            &mut workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(workspace_size)
}

/// Search for the fastest convolution forward algorithm
pub unsafe fn find_convolution_forward_algorithm(
    handle: &Handle,
    x_desc: &TensorDescriptor,
    x: *const c_void,
    w_desc: &TensorDescriptor,
    w: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    request_algo_count: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    exhaustive_search: bool,
) -> Result<(i32, Vec<ConvolutionPerf>)> {
    let mut returned_algo_count = 0;
    let mut perf_results = vec![unsafe { std::mem::zeroed() }; request_algo_count as usize];

    let status = unsafe {
        ffi::miopenFindConvolutionForwardAlgorithm(
            handle.as_raw(),
            x_desc.as_raw(),
            x,
            w_desc.as_raw(),
            w,
            conv_desc.as_raw(),
            y_desc.as_raw(),
            y,
            request_algo_count,
            &mut returned_algo_count,
            perf_results.as_mut_ptr(),
            workspace,
            workspace_size,
            exhaustive_search,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    perf_results.truncate(returned_algo_count as usize);
    Ok((returned_algo_count, perf_results))
}

/// Execute a forward convolution operation
pub unsafe fn convolution_forward(
    handle: &Handle,
    alpha: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    w_desc: &TensorDescriptor,
    w: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    algo: ConvFwdAlgorithm,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenConvolutionForward(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            x_desc.as_raw(),
            x,
            w_desc.as_raw(),
            w,
            conv_desc.as_raw(),
            algo,
            beta.as_ptr() as *const c_void,
            y_desc.as_raw(),
            y,
            workspace,
            workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Apply element-wise bias to a tensor
pub unsafe fn convolution_forward_bias(
    handle: &Handle,
    alpha: &[u8],
    b_desc: &TensorDescriptor,
    b: *const c_void,
    beta: &[u8],
    y_desc: &TensorDescriptor,
    y: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenConvolutionForwardBias(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            b_desc.as_raw(),
            b,
            beta.as_ptr() as *const c_void,
            y_desc.as_raw(),
            y,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Query the workspace size required for a backward data convolution algorithm
pub fn get_convolution_backward_data_workspace_size(
    handle: &Handle,
    dy_desc: &TensorDescriptor,
    w_desc: &TensorDescriptor,
    conv_desc: &ConvolutionDescriptor,
    dx_desc: &TensorDescriptor,
) -> Result<usize> {
    let mut workspace_size = 0;

    let status = unsafe {
        ffi::miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle.as_raw(),
            dy_desc.as_raw(),
            w_desc.as_raw(),
            conv_desc.as_raw(),
            dx_desc.as_raw(),
            &mut workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(workspace_size)
}

/// Search for the fastest convolution backward data algorithm
pub unsafe fn find_convolution_backward_data_algorithm(
    handle: &Handle,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    w_desc: &TensorDescriptor,
    w: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
    request_algo_count: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    exhaustive_search: bool,
) -> Result<(i32, Vec<ConvolutionPerf>)> {
    let mut returned_algo_count = 0;
    let mut perf_results = vec![unsafe { std::mem::zeroed() }; request_algo_count as usize];

    let status = unsafe {
        ffi::miopenFindConvolutionBackwardDataAlgorithm(
            handle.as_raw(),
            dy_desc.as_raw(),
            dy,
            w_desc.as_raw(),
            w,
            conv_desc.as_raw(),
            dx_desc.as_raw(),
            dx,
            request_algo_count,
            &mut returned_algo_count,
            perf_results.as_mut_ptr(),
            workspace,
            workspace_size,
            exhaustive_search,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    perf_results.truncate(returned_algo_count as usize);
    Ok((returned_algo_count, perf_results))
}

/// Execute a backward data convolution operation
pub unsafe fn convolution_backward_data(
    handle: &Handle,
    alpha: &[u8],
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    w_desc: &TensorDescriptor,
    w: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    algo: ConvBwdDataAlgorithm,
    beta: &[u8],
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenConvolutionBackwardData(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            dy_desc.as_raw(),
            dy,
            w_desc.as_raw(),
            w,
            conv_desc.as_raw(),
            algo,
            beta.as_ptr() as *const c_void,
            dx_desc.as_raw(),
            dx,
            workspace,
            workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Get the GPU memory required for the backward weights convolution algorithm
pub fn get_convolution_backward_weights_workspace_size(
    handle: &Handle,
    dy_desc: &TensorDescriptor,
    x_desc: &TensorDescriptor,
    conv_desc: &ConvolutionDescriptor,
    dw_desc: &TensorDescriptor,
) -> Result<usize> {
    let mut workspace_size = 0;

    let status = unsafe {
        ffi::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle.as_raw(),
            dy_desc.as_raw(),
            x_desc.as_raw(),
            conv_desc.as_raw(),
            dw_desc.as_raw(),
            &mut workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(workspace_size)
}

/// Search for the fastest convolution backward weights algorithm
pub unsafe fn find_convolution_backward_weights_algorithm(
    handle: &Handle,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    x_desc: &TensorDescriptor,
    x: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    dw_desc: &TensorDescriptor,
    dw: *mut c_void,
    request_algo_count: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    exhaustive_search: bool,
) -> Result<(i32, Vec<ConvolutionPerf>)> {
    let mut returned_algo_count = 0;
    let mut perf_results = vec![unsafe { std::mem::zeroed() }; request_algo_count as usize];

    let status = unsafe {
        ffi::miopenFindConvolutionBackwardWeightsAlgorithm(
            handle.as_raw(),
            dy_desc.as_raw(),
            dy,
            x_desc.as_raw(),
            x,
            conv_desc.as_raw(),
            dw_desc.as_raw(),
            dw,
            request_algo_count,
            &mut returned_algo_count,
            perf_results.as_mut_ptr(),
            workspace,
            workspace_size,
            exhaustive_search,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    perf_results.truncate(returned_algo_count as usize);
    Ok((returned_algo_count, perf_results))
}

/// Execute a backward weights convolution operation
pub unsafe fn convolution_backward_weights(
    handle: &Handle,
    alpha: &[u8],
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    x_desc: &TensorDescriptor,
    x: *const c_void,
    conv_desc: &ConvolutionDescriptor,
    algo: ConvBwdWeightsAlgorithm,
    beta: &[u8],
    dw_desc: &TensorDescriptor,
    dw: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenConvolutionBackwardWeights(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            dy_desc.as_raw(),
            dy,
            x_desc.as_raw(),
            x,
            conv_desc.as_raw(),
            algo,
            beta.as_ptr() as *const c_void,
            dw_desc.as_raw(),
            dw,
            workspace,
            workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Calculates the gradient with respect to the bias
pub unsafe fn convolution_backward_bias(
    handle: &Handle,
    alpha: &[u8],
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    beta: &[u8],
    db_desc: &TensorDescriptor,
    db: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenConvolutionBackwardBias(
            handle.as_raw(),
            alpha.as_ptr() as *const c_void,
            dy_desc.as_raw(),
            dy,
            beta.as_ptr() as *const c_void,
            db_desc.as_raw(),
            db,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}
