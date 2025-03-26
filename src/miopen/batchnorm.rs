// src/miopen/batchnorm.rs

use std::os::raw::c_void;
use crate::miopen::ffi;
use crate::miopen::error::{Error, Result};
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;

/// Batch normalization mode type
pub type BatchNormMode = ffi::miopenBatchNormMode_t;

/// Derive a tensor descriptor for batch normalization scale and bias
pub fn derive_bn_tensor_descriptor(
    derived_desc: &mut TensorDescriptor,
    x_desc: &TensorDescriptor,
    bn_mode: BatchNormMode,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenDeriveBNTensorDescriptor(
            derived_desc.as_raw(),
            x_desc.as_raw(),
            bn_mode,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization forward training
pub fn batch_normalization_forward_training(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha: &[u8],
    beta: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    bn_scale_bias_mean_var_desc: &TensorDescriptor,
    bn_scale: *mut c_void,
    bn_bias: *mut c_void,
    exp_avg_factor: f64,
    result_running_mean: *mut c_void,
    result_running_variance: *mut c_void,
    epsilon: f64,
    result_save_mean: *mut c_void,
    result_save_inv_variance: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationForwardTraining(
            handle.as_raw(),
            bn_mode,
            alpha.as_ptr() as *mut c_void,
            beta.as_ptr() as *mut c_void,
            x_desc.as_raw(),
            x,
            y_desc.as_raw(),
            y,
            bn_scale_bias_mean_var_desc.as_raw(),
            bn_scale,
            bn_bias,
            exp_avg_factor,
            result_running_mean,
            result_running_variance,
            epsilon,
            result_save_mean,
            result_save_inv_variance,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization forward training with separate tensor descriptors for scale, bias, mean, and variance
pub fn batch_normalization_forward_training_v2(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha: &[u8],
    beta: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    scale_desc: &TensorDescriptor,
    bias_desc: &TensorDescriptor,
    saved_mean_desc: &TensorDescriptor,
    saved_var_desc: &TensorDescriptor,
    bn_scale: *mut c_void,
    bn_bias: *mut c_void,
    exp_avg_factor: f64,
    result_running_mean: *mut c_void,
    result_running_variance: *mut c_void,
    epsilon: f64,
    result_save_mean: *mut c_void,
    result_save_inv_variance: *mut c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationForwardTraining_V2(
            handle.as_raw(),
            bn_mode,
            alpha.as_ptr() as *mut c_void,
            beta.as_ptr() as *mut c_void,
            x_desc.as_raw(),
            x,
            y_desc.as_raw(),
            y,
            scale_desc.as_raw(),
            bias_desc.as_raw(),
            saved_mean_desc.as_raw(),
            saved_var_desc.as_raw(),
            bn_scale,
            bn_bias,
            exp_avg_factor,
            result_running_mean,
            result_running_variance,
            epsilon,
            result_save_mean,
            result_save_inv_variance,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization forward inference
pub fn batch_normalization_forward_inference(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha: &[u8],
    beta: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    bn_scale_bias_mean_var_desc: &TensorDescriptor,
    bn_scale: *mut c_void,
    bn_bias: *mut c_void,
    estimated_mean: *mut c_void,
    estimated_variance: *mut c_void,
    epsilon: f64,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationForwardInference(
            handle.as_raw(),
            bn_mode,
            alpha.as_ptr() as *mut c_void,
            beta.as_ptr() as *mut c_void,
            x_desc.as_raw(),
            x,
            y_desc.as_raw(),
            y,
            bn_scale_bias_mean_var_desc.as_raw(),
            bn_scale,
            bn_bias,
            estimated_mean,
            estimated_variance,
            epsilon,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization forward inference with separate tensor descriptors for scale, bias, mean, and variance
pub fn batch_normalization_forward_inference_v2(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha: &[u8],
    beta: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    y_desc: &TensorDescriptor,
    y: *mut c_void,
    scale_desc: &TensorDescriptor,
    bias_desc: &TensorDescriptor,
    est_mean_desc: &TensorDescriptor,
    est_variance_desc: &TensorDescriptor,
    bn_scale: *mut c_void,
    bn_bias: *mut c_void,
    estimated_mean: *mut c_void,
    estimated_variance: *mut c_void,
    epsilon: f64,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationForwardInference_V2(
            handle.as_raw(),
            bn_mode,
            alpha.as_ptr() as *mut c_void,
            beta.as_ptr() as *mut c_void,
            x_desc.as_raw(),
            x,
            y_desc.as_raw(),
            y,
            scale_desc.as_raw(),
            bias_desc.as_raw(),
            est_mean_desc.as_raw(),
            est_variance_desc.as_raw(),
            bn_scale,
            bn_bias,
            estimated_mean,
            estimated_variance,
            epsilon,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization backward
pub fn batch_normalization_backward(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha_data_diff: &[u8],
    beta_data_diff: &[u8],
    alpha_param_diff: &[u8],
    beta_param_diff: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
    bn_scale_bias_diff_desc: &TensorDescriptor,
    bn_scale: *const c_void,
    result_bn_scale_diff: *mut c_void,
    result_bn_bias_diff: *mut c_void,
    epsilon: f64,
    saved_mean: *const c_void,
    saved_inv_variance: *const c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationBackward(
            handle.as_raw(),
            bn_mode,
            alpha_data_diff.as_ptr() as *const c_void,
            beta_data_diff.as_ptr() as *const c_void,
            alpha_param_diff.as_ptr() as *const c_void,
            beta_param_diff.as_ptr() as *const c_void,
            x_desc.as_raw(),
            x,
            dy_desc.as_raw(),
            dy,
            dx_desc.as_raw(),
            dx,
            bn_scale_bias_diff_desc.as_raw(),
            bn_scale,
            result_bn_scale_diff,
            result_bn_bias_diff,
            epsilon,
            saved_mean,
            saved_inv_variance,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute batch normalization backward with separate tensor descriptors for scale, bias, mean, and variance
pub fn batch_normalization_backward_v2(
    handle: &Handle,
    bn_mode: BatchNormMode,
    alpha_data_diff: &[u8],
    beta_data_diff: &[u8],
    alpha_param_diff: &[u8],
    beta_param_diff: &[u8],
    x_desc: &TensorDescriptor,
    x: *const c_void,
    dy_desc: &TensorDescriptor,
    dy: *const c_void,
    dx_desc: &TensorDescriptor,
    dx: *mut c_void,
    scale_desc: &TensorDescriptor,
    bias_desc: &TensorDescriptor,
    saved_mean_desc: &TensorDescriptor,
    saved_var_desc: &TensorDescriptor,
    bn_scale: *const c_void,
    result_bn_scale_diff: *mut c_void,
    result_bn_bias_diff: *mut c_void,
    epsilon: f64,
    saved_mean: *const c_void,
    saved_inv_variance: *const c_void,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenBatchNormalizationBackward_V2(
            handle.as_raw(),
            bn_mode,
            alpha_data_diff.as_ptr() as *const c_void,
            beta_data_diff.as_ptr() as *const c_void,
            alpha_param_diff.as_ptr() as *const c_void,
            beta_param_diff.as_ptr() as *const c_void,
            x_desc.as_raw(),
            x,
            dy_desc.as_raw(),
            dy,
            dx_desc.as_raw(),
            dx,
            scale_desc.as_raw(),
            bias_desc.as_raw(),
            saved_mean_desc.as_raw(),
            saved_var_desc.as_raw(),
            bn_scale,
            result_bn_scale_diff,
            result_bn_bias_diff,
            epsilon,
            saved_mean,
            saved_inv_variance,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}