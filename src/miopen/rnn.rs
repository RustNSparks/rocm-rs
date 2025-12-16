// src/miopen/rnn.rs

use crate::miopen::dropout::DropoutDescriptor;
use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use crate::miopen::handle::Handle;
use crate::miopen::tensor::TensorDescriptor;
use std::ptr;

/// RNN mode
pub type RNNMode = ffi::miopenRNNMode_t;

/// RNN input mode
pub type RNNInputMode = ffi::miopenRNNInputMode_t;

/// RNN algorithm
pub type RNNAlgo = ffi::miopenRNNAlgo_t;

/// RNN direction mode
pub type RNNDirectionMode = ffi::miopenRNNDirectionMode_t;

/// RNN bias mode
pub type RNNBiasMode = ffi::miopenRNNBiasMode_t;

/// Safe wrapper for MIOpen RNN descriptor
pub struct RNNDescriptor {
    desc: ffi::miopenRNNDescriptor_t,
}

impl RNNDescriptor {
    /// Create a new RNN descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateRNNDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the RNN descriptor
    pub fn set(
        &mut self,
        hidden_size: i32,
        num_layers: i32,
        input_mode: RNNInputMode,
        direction: RNNDirectionMode,
        mode: RNNMode,
        bias_mode: RNNBiasMode,
        algo: RNNAlgo,
        data_type: ffi::miopenDataType_t,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetRNNDescriptor(
                self.desc,
                hidden_size,
                num_layers,
                input_mode,
                direction,
                mode,
                bias_mode,
                algo,
                data_type,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Set the RNN descriptor with dropout
    pub fn set_with_dropout(
        &mut self,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: &DropoutDescriptor,
        input_mode: RNNInputMode,
        direction: RNNDirectionMode,
        mode: RNNMode,
        bias_mode: RNNBiasMode,
        algo: RNNAlgo,
        data_type: ffi::miopenDataType_t,
    ) -> Result<()> {
        let status = unsafe {
            ffi::miopenSetRNNDescriptor_V2(
                self.desc,
                hidden_size,
                num_layers,
                dropout_desc.as_raw(),
                input_mode,
                direction,
                mode,
                bias_mode,
                algo,
                data_type,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the RNN descriptor details
    pub fn get(
        &self,
    ) -> Result<(
        RNNMode,
        RNNAlgo,
        RNNInputMode,
        RNNDirectionMode,
        RNNBiasMode,
        i32,
        i32,
    )> {
        let mut mode = 0;
        let mut algo = 0;
        let mut input_mode = 0;
        let mut direction = 0;
        let mut bias_mode = 0;
        let mut hidden_size = 0;
        let mut num_layers = 0;

        let status = unsafe {
            ffi::miopenGetRNNDescriptor(
                self.desc,
                &mut mode,
                &mut algo,
                &mut input_mode,
                &mut direction,
                &mut bias_mode,
                &mut hidden_size,
                &mut num_layers,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok((
            mode,
            algo,
            input_mode,
            direction,
            bias_mode,
            hidden_size,
            num_layers,
        ))
    }

    /// Get the RNN descriptor details (version 2)
    pub fn get_v2(
        &self,
    ) -> Result<(
        i32,
        i32,
        Option<DropoutDescriptor>,
        RNNInputMode,
        RNNDirectionMode,
        RNNMode,
        RNNBiasMode,
        RNNAlgo,
        ffi::miopenDataType_t,
    )> {
        let mut hidden_size = 0;
        let mut num_layers = 0;
        let mut dropout_desc = ptr::null_mut();
        let mut input_mode = 0;
        let mut direction = 0;
        let mut mode = 0;
        let mut bias_mode = 0;
        let mut algo = 0;
        let mut data_type = 0;

        let status = unsafe {
            ffi::miopenGetRNNDescriptor_V2(
                self.desc,
                &mut hidden_size,
                &mut num_layers,
                &mut dropout_desc,
                &mut input_mode,
                &mut direction,
                &mut mode,
                &mut bias_mode,
                &mut algo,
                &mut data_type,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        // Note: We're creating a new DropoutDescriptor struct here, but we're not
        // taking ownership of the actual MIOpen object. It's still owned by the RNN descriptor.
        // This is a simplified approach for exposing the dropout descriptor.
        let dropout_descriptor = if dropout_desc.is_null() {
            None
        } else {
            Some(DropoutDescriptor::from_raw(dropout_desc))
        };

        Ok((
            hidden_size,
            num_layers,
            dropout_descriptor,
            input_mode,
            direction,
            mode,
            bias_mode,
            algo,
            data_type,
        ))
    }

    /// Set the RNN padding mode
    pub fn set_padding_mode(&mut self, padding_mode: ffi::miopenRNNPaddingMode_t) -> Result<()> {
        let status = unsafe { ffi::miopenSetRNNPaddingMode(self.desc, padding_mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the RNN padding mode
    pub fn get_padding_mode(&self) -> Result<ffi::miopenRNNPaddingMode_t> {
        let mut padding_mode = 0;

        let status = unsafe { ffi::miopenGetRNNPaddingMode(self.desc, &mut padding_mode) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(padding_mode)
    }

    /// Get workspace size for RNN
    pub fn get_workspace_size(
        &self,
        handle: &Handle,
        sequence_len: i32,
        x_desc: &[&TensorDescriptor],
    ) -> Result<usize> {
        let mut workspace_size = 0;

        let status = unsafe {
            ffi::miopenGetRNNWorkspaceSize(
                handle.as_raw(),
                self.desc,
                sequence_len,
                x_desc
                    .iter()
                    .map(|d| d.as_raw())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                &mut workspace_size,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(workspace_size)
    }

    /// Get reserve space size for RNN training
    pub fn get_training_reserve_size(
        &self,
        handle: &Handle,
        sequence_len: i32,
        x_desc: &[&TensorDescriptor],
    ) -> Result<usize> {
        let mut reserve_size = 0;

        let status = unsafe {
            ffi::miopenGetRNNTrainingReserveSize(
                handle.as_raw(),
                self.desc,
                sequence_len,
                x_desc
                    .iter()
                    .map(|d| d.as_raw())
                    .collect::<Vec<_>>()
                    .as_ptr(),
                &mut reserve_size,
            )
        };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(reserve_size)
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenRNNDescriptor_t {
        self.desc
    }
}

impl Drop for RNNDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe {
                let _ = ffi::miopenDestroyRNNDescriptor(self.desc);
                // We cannot handle errors in drop, so just ignore the result
            };
            self.desc = ptr::null_mut();
        }
    }
}

// We need to add a from_raw method to DropoutDescriptor to make the RNN wrapper work
impl DropoutDescriptor {
    // This method is only for internal use
    pub(crate) fn from_raw(desc: ffi::miopenDropoutDescriptor_t) -> Self {
        Self { desc }
    }
}

/// Execute forward inference for RNN
pub unsafe fn rnn_forward_inference(
    handle: &Handle,
    rnn_desc: &RNNDescriptor,
    sequence_len: i32,
    x_desc: &[&TensorDescriptor],
    x: *const std::os::raw::c_void,
    hx_desc: &TensorDescriptor,
    hx: *const std::os::raw::c_void,
    cx_desc: &TensorDescriptor,
    cx: *const std::os::raw::c_void,
    w_desc: &TensorDescriptor,
    w: *const std::os::raw::c_void,
    y_desc: &[&TensorDescriptor],
    y: *mut std::os::raw::c_void,
    hy_desc: &TensorDescriptor,
    hy: *mut std::os::raw::c_void,
    cy_desc: &TensorDescriptor,
    cy: *mut std::os::raw::c_void,
    workspace: *mut std::os::raw::c_void,
    workspace_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenRNNForwardInference(
            handle.as_raw(),
            rnn_desc.as_raw(),
            sequence_len,
            x_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            x,
            hx_desc.as_raw(),
            hx,
            cx_desc.as_raw(),
            cx,
            w_desc.as_raw(),
            w,
            y_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            y,
            hy_desc.as_raw(),
            hy,
            cy_desc.as_raw(),
            cy,
            workspace,
            workspace_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute forward training for RNN
pub unsafe fn rnn_forward_training(
    handle: &Handle,
    rnn_desc: &RNNDescriptor,
    sequence_len: i32,
    x_desc: &[&TensorDescriptor],
    x: *const std::os::raw::c_void,
    hx_desc: &TensorDescriptor,
    hx: *const std::os::raw::c_void,
    cx_desc: &TensorDescriptor,
    cx: *const std::os::raw::c_void,
    w_desc: &TensorDescriptor,
    w: *const std::os::raw::c_void,
    y_desc: &[&TensorDescriptor],
    y: *mut std::os::raw::c_void,
    hy_desc: &TensorDescriptor,
    hy: *mut std::os::raw::c_void,
    cy_desc: &TensorDescriptor,
    cy: *mut std::os::raw::c_void,
    workspace: *mut std::os::raw::c_void,
    workspace_size: usize,
    reserve_space: *mut std::os::raw::c_void,
    reserve_space_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenRNNForwardTraining(
            handle.as_raw(),
            rnn_desc.as_raw(),
            sequence_len,
            x_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            x,
            hx_desc.as_raw(),
            hx,
            cx_desc.as_raw(),
            cx,
            w_desc.as_raw(),
            w,
            y_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            y,
            hy_desc.as_raw(),
            hy,
            cy_desc.as_raw(),
            cy,
            workspace,
            workspace_size,
            reserve_space,
            reserve_space_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute backward data for RNN
pub unsafe fn rnn_backward_data(
    handle: &Handle,
    rnn_desc: &RNNDescriptor,
    sequence_len: i32,
    y_desc: &[&TensorDescriptor],
    y: *const std::os::raw::c_void,
    dy_desc: &[&TensorDescriptor],
    dy: *const std::os::raw::c_void,
    dhy_desc: &TensorDescriptor,
    dhy: *const std::os::raw::c_void,
    dcy_desc: &TensorDescriptor,
    dcy: *const std::os::raw::c_void,
    w_desc: &TensorDescriptor,
    w: *const std::os::raw::c_void,
    hx_desc: &TensorDescriptor,
    hx: *const std::os::raw::c_void,
    cx_desc: &TensorDescriptor,
    cx: *const std::os::raw::c_void,
    dx_desc: &[&TensorDescriptor],
    dx: *mut std::os::raw::c_void,
    dhx_desc: &TensorDescriptor,
    dhx: *mut std::os::raw::c_void,
    dcx_desc: &TensorDescriptor,
    dcx: *mut std::os::raw::c_void,
    workspace: *mut std::os::raw::c_void,
    workspace_size: usize,
    reserve_space: *mut std::os::raw::c_void,
    reserve_space_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenRNNBackwardData(
            handle.as_raw(),
            rnn_desc.as_raw(),
            sequence_len,
            y_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            y,
            dy_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            dy,
            dhy_desc.as_raw(),
            dhy,
            dcy_desc.as_raw(),
            dcy,
            w_desc.as_raw(),
            w,
            hx_desc.as_raw(),
            hx,
            cx_desc.as_raw(),
            cx,
            dx_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            dx,
            dhx_desc.as_raw(),
            dhx,
            dcx_desc.as_raw(),
            dcx,
            workspace,
            workspace_size,
            reserve_space,
            reserve_space_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}

/// Execute backward weights for RNN
pub unsafe fn rnn_backward_weights(
    handle: &Handle,
    rnn_desc: &RNNDescriptor,
    sequence_len: i32,
    x_desc: &[&TensorDescriptor],
    x: *const std::os::raw::c_void,
    hx_desc: &TensorDescriptor,
    hx: *const std::os::raw::c_void,
    y_desc: &[&TensorDescriptor],
    y: *const std::os::raw::c_void,
    dw_desc: &TensorDescriptor,
    dw: *mut std::os::raw::c_void,
    workspace: *mut std::os::raw::c_void,
    workspace_size: usize,
    reserve_space: *const std::os::raw::c_void,
    reserve_space_size: usize,
) -> Result<()> {
    let status = unsafe {
        ffi::miopenRNNBackwardWeights(
            handle.as_raw(),
            rnn_desc.as_raw(),
            sequence_len,
            x_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            x,
            hx_desc.as_raw(),
            hx,
            y_desc
                .iter()
                .map(|d| d.as_raw())
                .collect::<Vec<_>>()
                .as_ptr(),
            y,
            dw_desc.as_raw(),
            dw,
            workspace,
            workspace_size,
            reserve_space,
            reserve_space_size,
        )
    };

    if status != ffi::miopenStatus_t_miopenStatusSuccess {
        return Err(Error::new(status));
    }

    Ok(())
}
