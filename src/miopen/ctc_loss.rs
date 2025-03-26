// src/miopen/ctc_loss.rs

use std::ptr;
use std::os::raw::{c_void, c_int};
use crate::miopen::ffi;
use crate::miopen::error::{Error, Result};
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;

/// CTC Loss algorithm
pub type CTCLossAlgo = ffi::miopenCTCLossAlgo_t;

/// Safe wrapper for MIOpen CTC Loss descriptor
pub struct CTCLossDescriptor {
    desc: ffi::miopenCTCLossDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for CTCLossDescriptor {}
unsafe impl Sync for CTCLossDescriptor {}

impl CTCLossDescriptor {
    /// Create a new CTC Loss descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateCTCLossDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the CTC Loss descriptor
    pub fn set(&mut self, data_type: ffi::miopenDataType_t, blank_label_id: i32, apply_softmax_layer: bool) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetCTCLossDescriptor(
                self.desc,
                data_type,
                blank_label_id,
                apply_softmax_layer,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the CTC Loss descriptor details
    pub fn get(&self) -> Result<(ffi::miopenDataType_t, i32, bool)> {
        let mut data_type = 0;
        let mut blank_label_id = 0;
        let mut apply_softmax_layer = false;

        let status = unsafe {
            ffi::miopenGetCTCLossDescriptor(
                self.desc,
                &mut data_type,
                &mut blank_label_id,
                &mut apply_softmax_layer,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((data_type, blank_label_id, apply_softmax_layer))
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenCTCLossDescriptor_t {
        self.desc
    }
}

impl Drop for CTCLossDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyCTCLossDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

/// Get the workspace size required for CTC Loss operations
pub fn get_ctc_loss_workspace_size(
    handle: &Handle,
    probs_desc: &TensorDescriptor,
    gradients_desc: &TensorDescriptor,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    algo: CTCLossAlgo,
    ctc_loss_desc: &CTCLossDescriptor,
) -> Result<usize> {
    let mut workspace_size = 0;

    let status = unsafe {
        ffi::miopenGetCTCLossWorkspaceSize(
            handle.as_raw(),
            probs_desc.as_raw(),
            gradients_desc.as_raw(),
            labels.as_ptr(),
            label_lengths.as_ptr(),
            input_lengths.as_ptr(),
            algo,
            ctc_loss_desc.as_raw(),
            &mut workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(workspace_size)
}

/// Execute CTC Loss forward and gradient computation
pub fn ctc_loss(
    handle: &Handle,
    probs_desc: &TensorDescriptor,
    probs: *const c_void,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    losses: *mut c_void,
    gradients_desc: &TensorDescriptor,
    gradients: *mut c_void,
    algo: CTCLossAlgo,
    ctc_loss_desc: &CTCLossDescriptor,
    workspace: *mut c_void,
    workspace_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenCTCLoss(
            handle.as_raw(),
            probs_desc.as_raw(),
            probs,
            labels.as_ptr(),
            label_lengths.as_ptr(),
            input_lengths.as_ptr(),
            losses,
            gradients_desc.as_raw(),
            gradients,
            algo,
            ctc_loss_desc.as_raw(),
            workspace,
            workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}