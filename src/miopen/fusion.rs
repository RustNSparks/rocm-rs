// src/miopen/fusion.rs

use crate::miopen::activation::ActivationMode;
use crate::miopen::convolution::{ConvFwdAlgorithm, ConvolutionDescriptor};
use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::os::raw::c_void;
use std::ptr;

/// Fusion direction
pub type FusionDirection = ffi::miopenFusionDirection_t;

/// Safe wrapper for MIOpen fusion plan descriptor
pub struct FusionPlanDescriptor {
    desc: ffi::miopenFusionPlanDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for FusionPlanDescriptor {}
unsafe impl Sync for FusionPlanDescriptor {}

/// Safe wrapper for MIOpen fusion operator descriptor
pub struct FusionOpDescriptor {
    desc: ffi::miopenFusionOpDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for FusionOpDescriptor {}
unsafe impl Sync for FusionOpDescriptor {}

/// Safe wrapper for MIOpen operator arguments
pub struct OperatorArgs {
    args: ffi::miopenOperatorArgs_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for OperatorArgs {}
unsafe impl Sync for OperatorArgs {}

impl FusionPlanDescriptor {
    /// Create a new fusion plan descriptor
    pub fn new(fusion_direction: FusionDirection, input_desc: &TensorDescriptor) -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe {
            ffi::miopenCreateFusionPlan(&mut desc, fusion_direction, input_desc.as_raw())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Compile the fusion plan
    pub fn compile(&self, handle: &Handle) -> Result<()> {
        let status = unsafe { ffi::miopenCompileFusionPlan(handle.as_raw(), self.desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get an operator from the fusion plan
    pub fn get_op(&self, op_idx: i32) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe { ffi::miopenFusionPlanGetOp(self.desc, op_idx, &mut op) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Get the workspace size required for the fusion plan
    pub fn get_workspace_size(&self, handle: &Handle, algo: ConvFwdAlgorithm) -> Result<usize> {
        let mut workspace_size = 0;

        let status = unsafe {
            ffi::miopenFusionPlanGetWorkSpaceSize(
                handle.as_raw(),
                self.desc,
                &mut workspace_size,
                algo,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(workspace_size)
    }

    /// Create a forward convolution operator in the fusion plan
    pub fn create_op_conv_forward(
        &self,
        conv_desc: &ConvolutionDescriptor,
        w_desc: &TensorDescriptor,
    ) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe {
            ffi::miopenCreateOpConvForward(self.desc, &mut op, conv_desc.as_raw(), w_desc.as_raw())
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a forward activation operator in the fusion plan
    pub fn create_op_activation_forward(&self, mode: ActivationMode) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe { ffi::miopenCreateOpActivationForward(self.desc, &mut op, mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a backward activation operator in the fusion plan
    pub fn create_op_activation_backward(
        &self,
        mode: ActivationMode,
    ) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe { ffi::miopenCreateOpActivationBackward(self.desc, &mut op, mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a forward bias operator in the fusion plan
    pub fn create_op_bias_forward(&self, b_desc: &TensorDescriptor) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe { ffi::miopenCreateOpBiasForward(self.desc, &mut op, b_desc.as_raw()) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a batch normalization inference operator in the fusion plan
    pub fn create_op_batch_norm_inference(
        &self,
        bn_mode: ffi::miopenBatchNormMode_t,
        bn_scale_bias_mean_var_desc: &TensorDescriptor,
    ) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe {
            ffi::miopenCreateOpBatchNormInference(
                self.desc,
                &mut op,
                bn_mode,
                bn_scale_bias_mean_var_desc.as_raw(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a batch normalization forward operator in the fusion plan
    pub fn create_op_batch_norm_forward(
        &self,
        bn_mode: ffi::miopenBatchNormMode_t,
        running_mean_variance: bool,
    ) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe {
            ffi::miopenCreateOpBatchNormForward(self.desc, &mut op, bn_mode, running_mean_variance)
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Create a batch normalization backward operator in the fusion plan
    pub fn create_op_batch_norm_backward(
        &self,
        bn_mode: ffi::miopenBatchNormMode_t,
    ) -> Result<FusionOpDescriptor> {
        let mut op = ptr::null_mut();

        let status = unsafe { ffi::miopenCreateOpBatchNormBackward(self.desc, &mut op, bn_mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(FusionOpDescriptor { desc: op })
    }

    /// Get the convolution algorithms available for the fusion plan
    pub fn get_conv_algorithms(
        &self,
        request_algo_count: i32,
    ) -> Result<(i32, Vec<ConvFwdAlgorithm>)> {
        let mut returned_algo_count = 0;
        let mut algos = vec![0; request_algo_count as usize];

        let status = unsafe {
            ffi::miopenFusionPlanConvolutionGetAlgo(
                self.desc,
                request_algo_count,
                &mut returned_algo_count,
                algos.as_mut_ptr(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        algos.truncate(returned_algo_count as usize);
        Ok((returned_algo_count, algos))
    }

    /// Set the convolution algorithm for the fusion plan
    pub fn set_conv_algorithm(&self, algo: ConvFwdAlgorithm) -> Result<()> {
        let status = unsafe { ffi::miopenFusionPlanConvolutionSetAlgo(self.desc, algo) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Execute the fusion plan
    pub unsafe fn execute(
        &self,
        handle: &Handle,
        input_desc: &TensorDescriptor,
        input: *const c_void,
        output_desc: &TensorDescriptor,
        output: *mut c_void,
        args: &OperatorArgs,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenExecuteFusionPlan(
                handle.as_raw(),
                self.desc,
                input_desc.as_raw(),
                input,
                output_desc.as_raw(),
                output,
                args.as_raw(),
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenFusionPlanDescriptor_t {
        self.desc
    }
}

impl Drop for FusionPlanDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyFusionPlan(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

impl FusionOpDescriptor {
    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenFusionOpDescriptor_t {
        self.desc
    }
}

impl OperatorArgs {
    /// Create a new operator args
    pub fn new() -> Result<Self> {
        let mut args = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateOperatorArgs(&mut args) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { args })
    }

    /// Set arguments for a forward convolution op
    pub unsafe fn set_conv_forward(
        &self,
        conv_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        w: *const c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsConvForward(
                self.args,
                conv_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                w,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set arguments for a forward activation op
    pub unsafe fn set_activation_forward(
        &self,
        activ_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        activ_alpha: f64,
        activ_beta: f64,
        activ_gamma: f64,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsActivForward(
                self.args,
                activ_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                activ_alpha,
                activ_beta,
                activ_gamma,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set arguments for a backward activation op
    pub unsafe fn set_activation_backward(
        &self,
        activ_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        y: *const c_void,
        reserved: *const c_void,
        activ_alpha: f64,
        activ_beta: f64,
        activ_gamma: f64,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsActivBackward(
                self.args,
                activ_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                y,
                reserved,
                activ_alpha,
                activ_beta,
                activ_gamma,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set arguments for a batch normalization inference op
    pub unsafe fn set_batch_norm_inference(
        &self,
        bn_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        bn_scale: *const c_void,
        bn_bias: *const c_void,
        estimated_mean: *const c_void,
        estimated_variance: *const c_void,
        epsilon: f64,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsBatchNormInference(
                self.args,
                bn_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
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

    /// Set arguments for a batch normalization forward op
    pub unsafe fn set_batch_norm_forward(
        &self,
        bn_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        bn_scale: *const c_void,
        bn_bias: *const c_void,
        saved_mean: *mut c_void,
        saved_inv_variance: *mut c_void,
        running_mean: *mut c_void,
        running_variance: *mut c_void,
        exp_avg_factor: f64,
        epsilon: f64,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsBatchNormForward(
                self.args,
                bn_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                bn_scale,
                bn_bias,
                saved_mean,
                saved_inv_variance,
                running_mean,
                running_variance,
                exp_avg_factor,
                epsilon,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set arguments for a batch normalization backward op
    pub unsafe fn set_batch_norm_backward(
        &self,
        bn_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        x: *const c_void,
        bn_scale: *const c_void,
        bn_bias: *const c_void,
        result_bn_scale_diff: *mut c_void,
        result_bn_bias_diff: *mut c_void,
        saved_mean: *const c_void,
        saved_inv_variance: *const c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsBatchNormBackward(
                self.args,
                bn_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                x,
                bn_scale,
                bn_bias,
                result_bn_scale_diff,
                result_bn_bias_diff,
                saved_mean,
                saved_inv_variance,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set arguments for a bias forward op
    pub unsafe fn set_bias_forward(
        &self,
        bias_op: &FusionOpDescriptor,
        alpha: &[u8],
        beta: &[u8],
        bias: *const c_void,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetOpArgsBiasForward(
                self.args,
                bias_op.as_raw(),
                alpha.as_ptr() as *const c_void,
                beta.as_ptr() as *const c_void,
                bias,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the raw args
    pub fn as_raw(&self) -> ffi::miopenOperatorArgs_t {
        self.args
    }
}

impl Drop for OperatorArgs {
    fn drop(&mut self) {
        if !self.args.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyOperatorArgs(self.args);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.args = ptr::null_mut();
        }
    }
}
