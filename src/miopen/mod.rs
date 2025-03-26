// src/miopen/mod.rs

// Private modules
pub mod error;
pub mod handle;
pub mod tensor;
pub mod convolution;
pub mod pooling;
pub mod activation;
pub mod lrn;
pub mod batchnorm;
pub mod softmax;
pub mod dropout;
pub mod rnn;
pub mod reduce;
pub mod fusion;
pub mod mha;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
pub(crate) mod bindings;

// Public re-export of FFI for internal use
pub mod ffi;
pub mod ctc_loss;

// Re-export the main components for the public API
pub use error::{Error, Result};
pub use handle::Handle;
pub use tensor::{TensorDescriptor, SeqTensorDescriptor, DataType, TensorLayout};
pub use convolution::{ConvolutionDescriptor, ConvolutionMode,
                      ConvFwdAlgorithm, ConvBwdDataAlgorithm, ConvBwdWeightsAlgorithm,
                      ConvolutionPerf, find_convolution_forward_algorithm,
                      convolution_forward, convolution_backward_data, convolution_backward_weights};
pub use pooling::{PoolingDescriptor, PoolingMode, PoolingWorkspaceIndexMode};
pub use activation::{ActivationDescriptor, ActivationMode};
pub use lrn::{LRNDescriptor, LRNMode};
pub use batchnorm::BatchNormMode;
pub use softmax::{SoftmaxDescriptor, SoftmaxAlgorithm, SoftmaxMode,
                  softmax_forward, softmax_backward, softmax_forward_v2, softmax_backward_v2};
pub use dropout::{DropoutDescriptor, RNGType};
pub use rnn::{RNNDescriptor, RNNMode, RNNInputMode, RNNAlgo, RNNDirectionMode, RNNBiasMode};
pub use reduce::{ReduceTensorDescriptor, ReduceTensorOp, NanPropagation, ReduceTensorIndices, IndicesType};
pub use fusion::{FusionPlanDescriptor, FusionOpDescriptor, OperatorArgs, FusionDirection};

// New components
pub use mha::{MhaDescriptor, MhaMask, mha_mask, TensorArgumentId, tensor_argument_id};


/// Get MIOpen version information
pub fn get_version() -> Result<(usize, usize, usize)> {
    let mut major = 0;
    let mut minor = 0;
    let mut patch = 0;

    let status = unsafe {
        ffi::miopenGetVersion(&mut major, &mut minor, &mut patch)
    };

    Error::from_miopen_status_with_value(status, (major, minor, patch))
}