// src/miopen/mha.rs

use crate::miopen::error::{Error, Result};
use crate::miopen::ffi;
use std::ptr;

/// MHA mask mode
pub type MhaMask = ffi::miopenMhaMask_t;

/// Constants for MHA mask modes
pub mod mha_mask {
    use crate::miopen::ffi;

    /// No mask for MHA
    pub const NONE: super::MhaMask = ffi::miopenMhaMask_t_miopenMhaMaskNone;

    /// Causal mask for MHA
    pub const CAUSAL: super::MhaMask = ffi::miopenMhaMask_t_miopenMhaMaskCausal;
}

/// Safe wrapper for MIOpen MHA descriptor
pub struct MhaDescriptor {
    desc: ffi::miopenMhaDescriptor_t,
}

// Can't be automatically derived since we have a raw pointer
unsafe impl Send for MhaDescriptor {}
unsafe impl Sync for MhaDescriptor {}

impl MhaDescriptor {
    /// Create a new MHA descriptor
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        let status = unsafe { ffi::miopenCreateMhaDescriptor(&mut desc) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(Self { desc })
    }

    /// Set the MHA descriptor parameters
    pub fn set(&mut self, scale: f32) -> Result<()> {
        let status = unsafe { ffi::miopenSetMhaDescriptor(self.desc, scale) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(())
    }

    /// Get the MHA descriptor parameters
    pub fn get(&self) -> Result<f32> {
        let mut scale = 0.0f32;
        let status = unsafe { ffi::miopenGetMhaDescriptor(self.desc, &mut scale) };

        if status != ffi::miopenStatus_t_miopenStatusSuccess {
            return Err(Error::new(status));
        }

        Ok(scale)
    }

    /// Get the raw descriptor
    pub fn as_raw(&self) -> ffi::miopenMhaDescriptor_t {
        self.desc
    }
}

impl Drop for MhaDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            // No explicit destroy function in the API, assuming it's managed by the MIOpen context
        }
    }
}

/// Identifiers for tensor arguments of MHA problems
pub type TensorArgumentId = ffi::miopenTensorArgumentId_t;

/// Constants for tensor argument IDs
pub mod tensor_argument_id {
    use crate::miopen::ffi;

    // MHA tensor arguments
    pub const MHA_K: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaK;
    pub const MHA_Q: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaQ;
    pub const MHA_V: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaV;
    pub const MHA_O: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaO;
    pub const MHA_MASK: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaMask;
    pub const MHA_BIAS: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaBias;

    // Scale/descale tensors
    pub const MHA_DESCALE_K: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleK;
    pub const MHA_DESCALE_Q: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleQ;
    pub const MHA_DESCALE_V: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleV;
    pub const MHA_DESCALE_S: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleS;
    pub const MHA_SCALE_S: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleS;
    pub const MHA_SCALE_O: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleO;

    // Dropout related tensors
    pub const MHA_DROPOUT_PROBABILITY: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDropoutProbability;
    pub const MHA_DROPOUT_SEED: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDropoutSeed;
    pub const MHA_DROPOUT_OFFSET: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDropoutOffset;

    // Other MHA tensors
    pub const MHA_AMAX_O: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxO;
    pub const MHA_AMAX_S: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxS;
    pub const MHA_M: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaM;
    pub const MHA_Z_INV: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaZInv;

    // Backward tensors
    pub const MHA_DO: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaDO;
    pub const MHA_DESCALE_O: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleO;
    pub const MHA_DESCALE_DO: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleDO;
    pub const MHA_DESCALE_DS: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaDescaleDS;
    pub const MHA_SCALE_DS: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleDS;
    pub const MHA_SCALE_DQ: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleDQ;
    pub const MHA_SCALE_DK: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleDK;
    pub const MHA_SCALE_DV: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaScaleDV;
    pub const MHA_DQ: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaDQ;
    pub const MHA_DK: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaDK;
    pub const MHA_DV: super::TensorArgumentId = ffi::miopenTensorArgumentId_t_miopenTensorMhaDV;
    pub const MHA_AMAX_DQ: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxDQ;
    pub const MHA_AMAX_DK: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxDK;
    pub const MHA_AMAX_DV: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxDV;
    pub const MHA_AMAX_DS: super::TensorArgumentId =
        ffi::miopenTensorArgumentId_t_miopenTensorMhaAmaxDS;
}
