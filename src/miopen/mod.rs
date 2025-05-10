// src/miopen/mod.rs

// Private modules
pub mod activation;
pub mod batchnorm;
pub mod convolution;
pub mod dropout;
pub mod error;
pub mod fusion;
pub mod handle;
pub mod lrn;
pub mod mha;
pub mod pooling;
pub mod reduce;
pub mod rnn;
pub mod softmax;
pub mod tensor;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
#[allow(warnings)]
pub(crate) mod bindings;

// Public re-export of FFI for internal use
pub mod ctc_loss;
pub mod ffi;

// Re-export the main components for the public API
pub use activation::{ActivationDescriptor, ActivationMode};
pub use batchnorm::BatchNormMode;
pub use convolution::{
    ConvBwdDataAlgorithm, ConvBwdWeightsAlgorithm, ConvFwdAlgorithm, ConvolutionDescriptor,
    ConvolutionMode, ConvolutionPerf, convolution_backward_data, convolution_backward_weights,
    convolution_forward, find_convolution_forward_algorithm,
};
pub use dropout::{DropoutDescriptor, RNGType};
pub use error::{Error, Result};
pub use fusion::{FusionDirection, FusionOpDescriptor, FusionPlanDescriptor, OperatorArgs};
pub use handle::Handle;
pub use lrn::{LRNDescriptor, LRNMode};
pub use pooling::{PoolingDescriptor, PoolingMode, PoolingWorkspaceIndexMode};
pub use reduce::{
    IndicesType, NanPropagation, ReduceTensorDescriptor, ReduceTensorIndices, ReduceTensorOp,
};
pub use rnn::{RNNAlgo, RNNBiasMode, RNNDescriptor, RNNDirectionMode, RNNInputMode, RNNMode};
pub use softmax::{
    SoftmaxAlgorithm, SoftmaxDescriptor, SoftmaxMode, softmax_backward, softmax_backward_v2,
    softmax_forward, softmax_forward_v2,
};
pub use tensor::{DataType, SeqTensorDescriptor, TensorDescriptor, TensorLayout};

// New components
pub use mha::{MhaDescriptor, MhaMask, TensorArgumentId, mha_mask, tensor_argument_id};

/// Get MIOpen version information
pub fn get_version() -> Result<(usize, usize, usize)> {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;

    let status = unsafe { ffi::miopenGetVersion(&mut major, &mut minor, &mut patch) };

    Error::from_miopen_status_with_value(status, (major, minor, patch))
}
