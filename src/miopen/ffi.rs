// src/miopen/ffi.rs
//
// FFI bindings for the MIOpen API
// This file re-exports the necessary symbols from the auto-generated bindings

// We assume there's a bindings module that was auto-generated
// using bindgen or similar tool
use crate::miopen::bindings;

// Re-export the necessary types, constants, and functions

// Status and error handling
pub use bindings::miopenStatus_t;
pub use bindings::miopenStatus_t_miopenStatusSuccess;
pub use bindings::miopenStatus_t_miopenStatusNotInitialized;
pub use bindings::miopenStatus_t_miopenStatusInvalidValue;
pub use bindings::miopenStatus_t_miopenStatusBadParm;
pub use bindings::miopenStatus_t_miopenStatusAllocFailed;
pub use bindings::miopenStatus_t_miopenStatusInternalError;
pub use bindings::miopenStatus_t_miopenStatusNotImplemented;
pub use bindings::miopenStatus_t_miopenStatusUnknownError;
pub use bindings::miopenStatus_t_miopenStatusUnsupportedOp;
pub use bindings::miopenGetErrorString;

// Handle operations
pub use bindings::miopenHandle_t;
pub use bindings::miopenCreate;
pub use bindings::miopenCreateWithStream;
pub use bindings::miopenDestroy;
pub use bindings::miopenSetStream;
pub use bindings::miopenGetStream;
pub use bindings::miopenEnableProfiling;
pub use bindings::miopenGetKernelTime;
pub use bindings::miopenSetAllocator;
pub use bindings::miopenGetVersion;

// Tensor operations
pub use bindings::miopenTensorDescriptor_t;
pub use bindings::miopenCreateTensorDescriptor;
pub use bindings::miopenSet4dTensorDescriptor;
pub use bindings::miopenSet4dTensorDescriptorEx;
pub use bindings::miopenGet4dTensorDescriptor;
pub use bindings::miopenSetTensorDescriptor;
pub use bindings::miopenGetTensorDescriptorSize;
pub use bindings::miopenGetTensorDescriptor;
pub use bindings::miopenGetTensorNumBytes;
pub use bindings::miopenDestroyTensorDescriptor;
pub use bindings::miopenCreateSeqTensorDescriptor;
pub use bindings::miopenDestroySeqTensorDescriptor;
pub use bindings::miopenSetRNNDataSeqTensorDescriptor;
pub use bindings::miopenGetRNNDataSeqTensorDescriptor;

// DataType enum
pub use bindings::miopenDataType_t;
pub use bindings::miopenDataType_t_miopenHalf;
pub use bindings::miopenDataType_t_miopenFloat;
pub use bindings::miopenDataType_t_miopenInt32;
pub use bindings::miopenDataType_t_miopenInt8;
pub use bindings::miopenDataType_t_miopenBFloat16;
pub use bindings::miopenDataType_t_miopenDouble;
pub use bindings::miopenDataType_t_miopenInt64;

// TensorLayout enum
pub use bindings::miopenTensorLayout_t;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHW;
pub use bindings::miopenTensorLayout_t_miopenTensorNHWC;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWN;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHWc4;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHWc8;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWNc4;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWNc8;
pub use bindings::miopenTensorLayout_t_miopenTensorNCDHW;
pub use bindings::miopenTensorLayout_t_miopenTensorNDHWC;
pub use bindings::miopenSeqTensorDescriptor_t;

// Tensor operations
pub use bindings::miopenOpTensor;
pub use bindings::miopenSetTensor;
pub use bindings::miopenScaleTensor;
pub use bindings::miopenTransformTensor;

// TensorOp enum
pub use bindings::miopenTensorOp_t;
pub use bindings::miopenTensorOp_t_miopenTensorOpAdd;
pub use bindings::miopenTensorOp_t_miopenTensorOpMul;
pub use bindings::miopenTensorOp_t_miopenTensorOpMin;
pub use bindings::miopenTensorOp_t_miopenTensorOpMax;


// Convolution operations
pub use bindings::miopenConvolutionDescriptor_t;
pub use bindings::miopenCreateConvolutionDescriptor;
pub use bindings::miopenInitConvolutionDescriptor;
pub use bindings::miopenInitConvolutionNdDescriptor;
pub use bindings::miopenGetConvolutionDescriptor;
pub use bindings::miopenGetConvolutionNdDescriptor;
pub use bindings::miopenGetConvolutionForwardOutputDim;
pub use bindings::miopenGetConvolutionNdForwardOutputDim;
pub use bindings::miopenDestroyConvolutionDescriptor;
pub use bindings::miopenConvolutionForwardGetWorkSpaceSize;
pub use bindings::miopenConvolutionForwardGetSolution;
pub use bindings::miopenConvolutionForwardGetSolutionCount;
pub use bindings::miopenFindConvolutionForwardAlgorithm;
pub use bindings::miopenConvolutionForward;
pub use bindings::miopenConvolutionBackwardDataGetWorkSpaceSize;
pub use bindings::miopenFindConvolutionBackwardDataAlgorithm;
pub use bindings::miopenConvolutionBackwardData;
pub use bindings::miopenConvolutionBackwardWeightsGetWorkSpaceSize;
pub use bindings::miopenFindConvolutionBackwardWeightsAlgorithm;
pub use bindings::miopenConvolutionBackwardWeights;
pub use bindings::miopenConvolutionForwardBias;
pub use bindings::miopenConvolutionBackwardBias;
pub use bindings::miopenSetConvolutionGroupCount;
pub use bindings::miopenGetConvolutionGroupCount;
pub use bindings::miopenSetConvolutionAttribute;
pub use bindings::miopenGetConvolutionAttribute;

// Convolution enums and structs
pub use bindings::miopenConvolutionMode_t;
pub use bindings::miopenConvolutionMode_t_miopenConvolution;
pub use bindings::miopenConvolutionMode_t_miopenTranspose;
pub use bindings::miopenConvolutionMode_t_miopenGroupConv;
pub use bindings::miopenConvolutionMode_t_miopenDepthwise;

pub use bindings::miopenConvFwdAlgorithm_t;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoGEMM;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoDirect;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoFFT;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoWinograd;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoImplicitGEMM;

pub use bindings::miopenConvBwdDataAlgorithm_t;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoGEMM;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoDirect;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoFFT;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoWinograd;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoImplicitGEMM;

pub use bindings::miopenConvBwdWeightsAlgorithm_t;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoGEMM;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoDirect;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoWinograd;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoImplicitGEMM;

pub use bindings::miopenPaddingMode_t;
pub use bindings::miopenPaddingMode_t_miopenPaddingDefault;
pub use bindings::miopenPaddingMode_t_miopenPaddingSame;
pub use bindings::miopenPaddingMode_t_miopenPaddingValid;

pub use bindings::miopenConvolutionAttrib_t;
pub use bindings::miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL;
pub use bindings::miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC;

pub use bindings::miopenConvAlgoPerf_t;
pub use bindings::miopenConvSolution_t;

// Pooling operations
pub use bindings::miopenPoolingDescriptor_t;
pub use bindings::miopenCreatePoolingDescriptor;
pub use bindings::miopenSet2dPoolingDescriptor;
pub use bindings::miopenSetNdPoolingDescriptor;
pub use bindings::miopenGet2dPoolingDescriptor;
pub use bindings::miopenGetNdPoolingDescriptor;
pub use bindings::miopenGetPoolingForwardOutputDim;
pub use bindings::miopenGetPoolingNdForwardOutputDim;
pub use bindings::miopenDestroyPoolingDescriptor;
pub use bindings::miopenPoolingGetWorkSpaceSize;
pub use bindings::miopenPoolingGetWorkSpaceSizeV2;
pub use bindings::miopenPoolingForward;
pub use bindings::miopenPoolingBackward;
pub use bindings::miopenSetPoolingIndexType;
pub use bindings::miopenGetPoolingIndexType;
pub use bindings::miopenSetPoolingWorkSpaceIndexMode;
pub use bindings::miopenGetPoolingWorkSpaceIndexMode;

// Pooling enums
pub use bindings::miopenPoolingMode_t;
pub use bindings::miopenPoolingMode_t_miopenPoolingMax;
pub use bindings::miopenPoolingMode_t_miopenPoolingAverage;
pub use bindings::miopenPoolingMode_t_miopenPoolingAverageInclusive;

pub use bindings::miopenPoolingWorkspaceIndexMode_t;
pub use bindings::miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexMask;
pub use bindings::miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexImage;

pub use bindings::miopenIndexType_t;
pub use bindings::miopenIndexType_t_miopenIndexUint8;
pub use bindings::miopenIndexType_t_miopenIndexUint16;
pub use bindings::miopenIndexType_t_miopenIndexUint32;
pub use bindings::miopenIndexType_t_miopenIndexUint64;

// LRN operations
pub use bindings::miopenLRNDescriptor_t;
pub use bindings::miopenCreateLRNDescriptor;
pub use bindings::miopenSetLRNDescriptor;
pub use bindings::miopenGetLRNDescriptor;
pub use bindings::miopenLRNGetWorkSpaceSize;
pub use bindings::miopenLRNForward;
pub use bindings::miopenLRNBackward;
pub use bindings::miopenDestroyLRNDescriptor;

// LRN enums
pub use bindings::miopenLRNMode_t;
pub use bindings::miopenLRNMode_t_miopenLRNWithinChannel;
pub use bindings::miopenLRNMode_t_miopenLRNCrossChannel;

// BatchNorm operations
pub use bindings::miopenDeriveBNTensorDescriptor;
pub use bindings::miopenBatchNormalizationForwardTraining;
pub use bindings::miopenBatchNormalizationForwardInference;
pub use bindings::miopenBatchNormalizationBackward;
pub use bindings::miopenBatchNormalizationForwardTraining_V2;
pub use bindings::miopenBatchNormalizationForwardInference_V2;
pub use bindings::miopenBatchNormalizationBackward_V2;

// BatchNorm enums
pub use bindings::miopenBatchNormMode_t;
pub use bindings::miopenBatchNormMode_t_miopenBNPerActivation;
pub use bindings::miopenBatchNormMode_t_miopenBNSpatial;

// Activation operations
pub use bindings::miopenActivationDescriptor_t;
pub use bindings::miopenCreateActivationDescriptor;
pub use bindings::miopenSetActivationDescriptor;
pub use bindings::miopenGetActivationDescriptor;
pub use bindings::miopenActivationForward;
pub use bindings::miopenActivationBackward;
pub use bindings::miopenDestroyActivationDescriptor;

// Activation enums
pub use bindings::miopenActivationMode_t;
pub use bindings::miopenActivationMode_t_miopenActivationPASTHRU;
pub use bindings::miopenActivationMode_t_miopenActivationLOGISTIC;
pub use bindings::miopenActivationMode_t_miopenActivationTANH;
pub use bindings::miopenActivationMode_t_miopenActivationRELU;
pub use bindings::miopenActivationMode_t_miopenActivationSOFTRELU;
pub use bindings::miopenActivationMode_t_miopenActivationABS;
pub use bindings::miopenActivationMode_t_miopenActivationPOWER;
pub use bindings::miopenActivationMode_t_miopenActivationCLIPPEDRELU;
pub use bindings::miopenActivationMode_t_miopenActivationLEAKYRELU;
pub use bindings::miopenActivationMode_t_miopenActivationELU;

// Softmax operations
pub use bindings::miopenSoftmaxDescriptor_t;
pub use bindings::miopenCreateSoftmaxDescriptor;
pub use bindings::miopenSetSoftmaxDescriptor;
pub use bindings::miopenGetSoftmaxDescriptor;
pub use bindings::miopenSoftmaxForward;
pub use bindings::miopenSoftmaxBackward;
pub use bindings::miopenSoftmaxForward_V2;
pub use bindings::miopenSoftmaxBackward_V2;

// Softmax enums
pub use bindings::miopenSoftmaxAlgorithm_t;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_FAST;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_ACCURATE;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_LOG;

pub use bindings::miopenSoftmaxMode_t;
pub use bindings::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_INSTANCE;
pub use bindings::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_CHANNEL;

// Fusion operations
pub use bindings::miopenFusionPlanDescriptor_t;
pub use bindings::miopenCreateFusionPlan;
pub use bindings::miopenDestroyFusionPlan;
pub use bindings::miopenCompileFusionPlan;
pub use bindings::miopenFusionPlanGetOp;
pub use bindings::miopenFusionPlanGetWorkSpaceSize;
pub use bindings::miopenFusionPlanConvolutionGetAlgo;
pub use bindings::miopenFusionPlanConvolutionSetAlgo;
pub use bindings::miopenCreateOpConvForward;
pub use bindings::miopenCreateOpActivationForward;
pub use bindings::miopenCreateOpActivationBackward;
pub use bindings::miopenCreateOpBiasForward;
pub use bindings::miopenCreateOpBatchNormInference;
pub use bindings::miopenCreateOpBatchNormForward;
pub use bindings::miopenCreateOpBatchNormBackward;
pub use bindings::miopenCreateOperatorArgs;
pub use bindings::miopenDestroyOperatorArgs;
pub use bindings::miopenSetOpArgsConvForward;
pub use bindings::miopenSetOpArgsActivForward;
pub use bindings::miopenSetOpArgsActivBackward;
pub use bindings::miopenSetOpArgsBatchNormInference;
pub use bindings::miopenSetOpArgsBatchNormForward;
pub use bindings::miopenSetOpArgsBatchNormBackward;
pub use bindings::miopenSetOpArgsBiasForward;
pub use bindings::miopenExecuteFusionPlan;
pub use bindings::miopenFusionOpDescriptor_t;
pub use bindings::miopenOperatorArgs_t;

// Fusion enums
pub use bindings::miopenFusionDirection_t;
pub use bindings::miopenFusionDirection_t_miopenVerticalFusion;
pub use bindings::miopenFusionDirection_t_miopenHorizontalFusion;

// RNN operations
pub use bindings::miopenRNNDescriptor_t;
pub use bindings::miopenCreateRNNDescriptor;
pub use bindings::miopenGetRNNDescriptor;
pub use bindings::miopenGetRNNDescriptor_V2;
pub use bindings::miopenDestroyRNNDescriptor;
pub use bindings::miopenSetRNNDescriptor;
pub use bindings::miopenSetRNNDescriptor_V2;
pub use bindings::miopenGetRNNWorkspaceSize;
pub use bindings::miopenGetRNNTrainingReserveSize;
pub use bindings::miopenGetRNNTempSpaceSizes;
pub use bindings::miopenGetRNNParamsSize;
pub use bindings::miopenGetRNNInputTensorSize;
pub use bindings::miopenGetRNNHiddenTensorSize;
pub use bindings::miopenGetRNNLayerParamSize;
pub use bindings::miopenGetRNNLayerBiasSize;
pub use bindings::miopenGetRNNLayerParam;
pub use bindings::miopenGetRNNLayerBias;
pub use bindings::miopenGetRNNLayerParamOffset;
pub use bindings::miopenGetRNNLayerBiasOffset;
pub use bindings::miopenSetRNNLayerParam;
pub use bindings::miopenSetRNNLayerBias;
pub use bindings::miopenRNNForward;
pub use bindings::miopenRNNBackwardData;
pub use bindings::miopenRNNBackwardWeights;
pub use bindings::miopenRNNForwardTraining;
pub use bindings::miopenRNNForwardInference;
pub use bindings::miopenAllocatorFunction;
pub use bindings::miopenDeallocatorFunction;
pub use bindings::miopenRNNBaseLayout_t;

// RNN enums
pub use bindings::miopenRNNMode_t;
pub use bindings::miopenRNNMode_t_miopenRNNRELU;
pub use bindings::miopenRNNMode_t_miopenRNNTANH;
pub use bindings::miopenRNNMode_t_miopenLSTM;
pub use bindings::miopenRNNMode_t_miopenGRU;

pub use bindings::miopenRNNInputMode_t;
pub use bindings::miopenRNNInputMode_t_miopenRNNlinear;
pub use bindings::miopenRNNInputMode_t_miopenRNNskip;

pub use bindings::miopenRNNAlgo_t;
pub use bindings::miopenRNNAlgo_t_miopenRNNdefault;
pub use bindings::miopenRNNAlgo_t_miopenRNNfundamental;

pub use bindings::miopenRNNDirectionMode_t;
pub use bindings::miopenRNNDirectionMode_t_miopenRNNunidirection;
pub use bindings::miopenRNNDirectionMode_t_miopenRNNbidirection;

pub use bindings::miopenRNNBiasMode_t;
pub use bindings::miopenRNNBiasMode_t_miopenRNNNoBias;
pub use bindings::miopenRNNBiasMode_t_miopenRNNwithBias;

pub use bindings::miopenRNNPaddingMode_t;
pub use bindings::miopenRNNPaddingMode_t_miopenRNNIONotPadded;
pub use bindings::miopenRNNPaddingMode_t_miopenRNNIOWithPadding;

pub use bindings::miopenRNNFWDMode_t;
pub use bindings::miopenRNNFWDMode_t_miopenRNNTraining;
pub use bindings::miopenRNNFWDMode_t_miopenRNNInference;

// Dropout operations
pub use bindings::miopenDropoutDescriptor_t;
pub use bindings::miopenCreateDropoutDescriptor;
pub use bindings::miopenDestroyDropoutDescriptor;
pub use bindings::miopenSetDropoutDescriptor;
pub use bindings::miopenGetDropoutDescriptor;
pub use bindings::miopenRestoreDropoutDescriptor;
pub use bindings::miopenDropoutGetReserveSpaceSize;
pub use bindings::miopenDropoutGetStatesSize;
pub use bindings::miopenDropoutForward;
pub use bindings::miopenDropoutBackward;

// Dropout enums
pub use bindings::miopenRNGType_t;
pub use bindings::miopenRNGType_t_MIOPEN_RNG_PSEUDO_XORWOW;

// Tensor reduction operations
pub use bindings::miopenReduceTensorDescriptor_t;
pub use bindings::miopenCreateReduceTensorDescriptor;
pub use bindings::miopenDestroyReduceTensorDescriptor;
pub use bindings::miopenSetReduceTensorDescriptor;
pub use bindings::miopenGetReduceTensorDescriptor;
pub use bindings::miopenGetReductionIndicesSize;
pub use bindings::miopenGetReductionWorkspaceSize;
pub use bindings::miopenReduceTensor;

// Tensor reduction enums
pub use bindings::miopenReduceTensorOp_t;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_ADD;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MUL;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MIN;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MAX;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AMAX;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AVG;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM1;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM2;

pub use bindings::miopenNanPropagation_t;
pub use bindings::miopenNanPropagation_t_MIOPEN_NOT_PROPAGATE_NAN;
pub use bindings::miopenNanPropagation_t_MIOPEN_PROPAGATE_NAN;

pub use bindings::miopenReduceTensorIndices_t;
pub use bindings::miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_NO_INDICES;
pub use bindings::miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES;

pub use bindings::miopenIndicesType_t;
pub use bindings::miopenIndicesType_t_MIOPEN_32BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_64BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_16BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_8BIT_INDICES;
pub use bindings::miopenSetNdTensorDescriptorWithLayout;
pub use bindings::miopenGetConvolutionSpatialDim;
pub use bindings::miopenSetTransposeConvOutputPadding;
pub use bindings::miopenSetTransposeConvNdOutputPadding;
pub use bindings::miopenGetRNNPaddingMode;
pub use bindings::miopenSetRNNPaddingMode;
// Other needed functions and types
// Add more as needed...