// src/miopen/ffi.rs
//
// FFI bindings for the MIOpen API
// This file re-exports the necessary symbols from the auto-generated bindings

// We assume there's a bindings module that was auto-generated
// using bindgen or similar tool
use crate::miopen::bindings;

// Re-export the necessary types, constants, and functions

// Status and error handling
pub use bindings::miopenGetErrorString;
pub use bindings::miopenStatus_t;
pub use bindings::miopenStatus_t_miopenStatusAllocFailed;
pub use bindings::miopenStatus_t_miopenStatusBadParm;
pub use bindings::miopenStatus_t_miopenStatusInternalError;
pub use bindings::miopenStatus_t_miopenStatusInvalidValue;
pub use bindings::miopenStatus_t_miopenStatusNotImplemented;
pub use bindings::miopenStatus_t_miopenStatusNotInitialized;
pub use bindings::miopenStatus_t_miopenStatusSuccess;
pub use bindings::miopenStatus_t_miopenStatusUnknownError;
pub use bindings::miopenStatus_t_miopenStatusUnsupportedOp;

// Handle operations
pub use bindings::miopenCreate;
pub use bindings::miopenCreateWithStream;
pub use bindings::miopenDestroy;
pub use bindings::miopenEnableProfiling;
pub use bindings::miopenGetKernelTime;
pub use bindings::miopenGetStream;
pub use bindings::miopenGetVersion;
pub use bindings::miopenHandle_t;
pub use bindings::miopenSetAllocator;
pub use bindings::miopenSetStream;

// Tensor operations
pub use bindings::miopenCreateSeqTensorDescriptor;
pub use bindings::miopenCreateTensorDescriptor;
pub use bindings::miopenDestroySeqTensorDescriptor;
pub use bindings::miopenDestroyTensorDescriptor;
pub use bindings::miopenGet4dTensorDescriptor;
pub use bindings::miopenGetRNNDataSeqTensorDescriptor;
pub use bindings::miopenGetTensorDescriptor;
pub use bindings::miopenGetTensorDescriptorSize;
pub use bindings::miopenGetTensorNumBytes;
pub use bindings::miopenSet4dTensorDescriptor;
pub use bindings::miopenSet4dTensorDescriptorEx;
pub use bindings::miopenSetRNNDataSeqTensorDescriptor;
pub use bindings::miopenSetTensorDescriptor;
pub use bindings::miopenTensorDescriptor_t;

// DataType enum
pub use bindings::miopenDataType_t;
pub use bindings::miopenDataType_t_miopenBFloat16;
pub use bindings::miopenDataType_t_miopenDouble;
pub use bindings::miopenDataType_t_miopenFloat;
pub use bindings::miopenDataType_t_miopenHalf;
pub use bindings::miopenDataType_t_miopenInt8;
pub use bindings::miopenDataType_t_miopenInt32;
pub use bindings::miopenDataType_t_miopenInt64;

// TensorLayout enum
pub use bindings::miopenSeqTensorDescriptor_t;
pub use bindings::miopenTensorLayout_t;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWN;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWNc4;
pub use bindings::miopenTensorLayout_t_miopenTensorCHWNc8;
pub use bindings::miopenTensorLayout_t_miopenTensorNCDHW;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHW;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHWc4;
pub use bindings::miopenTensorLayout_t_miopenTensorNCHWc8;
pub use bindings::miopenTensorLayout_t_miopenTensorNDHWC;
pub use bindings::miopenTensorLayout_t_miopenTensorNHWC;

// Tensor operations
pub use bindings::miopenOpTensor;
pub use bindings::miopenScaleTensor;
pub use bindings::miopenSetTensor;
pub use bindings::miopenTransformTensor;

// TensorOp enum
pub use bindings::miopenTensorOp_t;
pub use bindings::miopenTensorOp_t_miopenTensorOpAdd;
pub use bindings::miopenTensorOp_t_miopenTensorOpMax;
pub use bindings::miopenTensorOp_t_miopenTensorOpMin;
pub use bindings::miopenTensorOp_t_miopenTensorOpMul;

// Convolution operations
pub use bindings::miopenConvolutionBackwardBias;
pub use bindings::miopenConvolutionBackwardData;
pub use bindings::miopenConvolutionBackwardDataGetWorkSpaceSize;
pub use bindings::miopenConvolutionBackwardWeights;
pub use bindings::miopenConvolutionBackwardWeightsGetWorkSpaceSize;
pub use bindings::miopenConvolutionDescriptor_t;
pub use bindings::miopenConvolutionForward;
pub use bindings::miopenConvolutionForwardBias;
pub use bindings::miopenConvolutionForwardGetSolution;
pub use bindings::miopenConvolutionForwardGetSolutionCount;
pub use bindings::miopenConvolutionForwardGetWorkSpaceSize;
pub use bindings::miopenCreateConvolutionDescriptor;
pub use bindings::miopenDestroyConvolutionDescriptor;
pub use bindings::miopenFindConvolutionBackwardDataAlgorithm;
pub use bindings::miopenFindConvolutionBackwardWeightsAlgorithm;
pub use bindings::miopenFindConvolutionForwardAlgorithm;
pub use bindings::miopenGetConvolutionAttribute;
pub use bindings::miopenGetConvolutionDescriptor;
pub use bindings::miopenGetConvolutionForwardOutputDim;
pub use bindings::miopenGetConvolutionGroupCount;
pub use bindings::miopenGetConvolutionNdDescriptor;
pub use bindings::miopenGetConvolutionNdForwardOutputDim;
pub use bindings::miopenInitConvolutionDescriptor;
pub use bindings::miopenInitConvolutionNdDescriptor;
pub use bindings::miopenSetConvolutionAttribute;
pub use bindings::miopenSetConvolutionGroupCount;

// Convolution enums and structs
pub use bindings::miopenConvolutionMode_t;
pub use bindings::miopenConvolutionMode_t_miopenConvolution;
pub use bindings::miopenConvolutionMode_t_miopenDepthwise;
pub use bindings::miopenConvolutionMode_t_miopenGroupConv;
pub use bindings::miopenConvolutionMode_t_miopenTranspose;

pub use bindings::miopenConvFwdAlgorithm_t;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoDirect;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoFFT;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoGEMM;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoImplicitGEMM;
pub use bindings::miopenConvFwdAlgorithm_t_miopenConvolutionFwdAlgoWinograd;

pub use bindings::miopenConvBwdDataAlgorithm_t;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoDirect;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoFFT;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoGEMM;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoImplicitGEMM;
pub use bindings::miopenConvBwdDataAlgorithm_t_miopenConvolutionBwdDataAlgoWinograd;

pub use bindings::miopenConvBwdWeightsAlgorithm_t;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoDirect;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoGEMM;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoImplicitGEMM;
pub use bindings::miopenConvBwdWeightsAlgorithm_t_miopenConvolutionBwdWeightsAlgoWinograd;

pub use bindings::miopenPaddingMode_t;
pub use bindings::miopenPaddingMode_t_miopenPaddingDefault;
pub use bindings::miopenPaddingMode_t_miopenPaddingSame;
pub use bindings::miopenPaddingMode_t_miopenPaddingValid;

pub use bindings::miopenConvolutionAttrib_t;
pub use bindings::miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC;
pub use bindings::miopenConvolutionAttrib_t_MIOPEN_CONVOLUTION_ATTRIB_FP16_ALT_IMPL;

pub use bindings::miopenConvAlgoPerf_t;
pub use bindings::miopenConvSolution_t;

// Pooling operations
pub use bindings::miopenCreatePoolingDescriptor;
pub use bindings::miopenDestroyPoolingDescriptor;
pub use bindings::miopenGet2dPoolingDescriptor;
pub use bindings::miopenGetNdPoolingDescriptor;
pub use bindings::miopenGetPoolingForwardOutputDim;
pub use bindings::miopenGetPoolingIndexType;
pub use bindings::miopenGetPoolingNdForwardOutputDim;
pub use bindings::miopenGetPoolingWorkSpaceIndexMode;
pub use bindings::miopenPoolingBackward;
pub use bindings::miopenPoolingDescriptor_t;
pub use bindings::miopenPoolingForward;
pub use bindings::miopenPoolingGetWorkSpaceSize;
pub use bindings::miopenPoolingGetWorkSpaceSizeV2;
pub use bindings::miopenSet2dPoolingDescriptor;
pub use bindings::miopenSetNdPoolingDescriptor;
pub use bindings::miopenSetPoolingIndexType;
pub use bindings::miopenSetPoolingWorkSpaceIndexMode;

// Pooling enums
pub use bindings::miopenPoolingMode_t;
pub use bindings::miopenPoolingMode_t_miopenPoolingAverage;
pub use bindings::miopenPoolingMode_t_miopenPoolingAverageInclusive;
pub use bindings::miopenPoolingMode_t_miopenPoolingMax;

pub use bindings::miopenPoolingWorkspaceIndexMode_t;
pub use bindings::miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexImage;
pub use bindings::miopenPoolingWorkspaceIndexMode_t_miopenPoolingWorkspaceIndexMask;

pub use bindings::miopenIndexType_t;
pub use bindings::miopenIndexType_t_miopenIndexUint8;
pub use bindings::miopenIndexType_t_miopenIndexUint16;
pub use bindings::miopenIndexType_t_miopenIndexUint32;
pub use bindings::miopenIndexType_t_miopenIndexUint64;

// LRN operations
pub use bindings::miopenCreateLRNDescriptor;
pub use bindings::miopenDestroyLRNDescriptor;
pub use bindings::miopenGetLRNDescriptor;
pub use bindings::miopenLRNBackward;
pub use bindings::miopenLRNDescriptor_t;
pub use bindings::miopenLRNForward;
pub use bindings::miopenLRNGetWorkSpaceSize;
pub use bindings::miopenSetLRNDescriptor;

// LRN enums
pub use bindings::miopenLRNMode_t;
pub use bindings::miopenLRNMode_t_miopenLRNCrossChannel;
pub use bindings::miopenLRNMode_t_miopenLRNWithinChannel;

// BatchNorm operations
pub use bindings::miopenBatchNormalizationBackward;
pub use bindings::miopenBatchNormalizationBackward_V2;
pub use bindings::miopenBatchNormalizationForwardInference;
pub use bindings::miopenBatchNormalizationForwardInference_V2;
pub use bindings::miopenBatchNormalizationForwardTraining;
pub use bindings::miopenBatchNormalizationForwardTraining_V2;
pub use bindings::miopenDeriveBNTensorDescriptor;

// BatchNorm enums
pub use bindings::miopenBatchNormMode_t;
pub use bindings::miopenBatchNormMode_t_miopenBNPerActivation;
pub use bindings::miopenBatchNormMode_t_miopenBNSpatial;

// Activation operations
pub use bindings::miopenActivationBackward;
pub use bindings::miopenActivationDescriptor_t;
pub use bindings::miopenActivationForward;
pub use bindings::miopenCreateActivationDescriptor;
pub use bindings::miopenDestroyActivationDescriptor;
pub use bindings::miopenGetActivationDescriptor;
pub use bindings::miopenSetActivationDescriptor;

// Activation enums
pub use bindings::miopenActivationMode_t;
pub use bindings::miopenActivationMode_t_miopenActivationABS;
pub use bindings::miopenActivationMode_t_miopenActivationCLIPPEDRELU;
pub use bindings::miopenActivationMode_t_miopenActivationELU;
pub use bindings::miopenActivationMode_t_miopenActivationLEAKYRELU;
pub use bindings::miopenActivationMode_t_miopenActivationLOGISTIC;
pub use bindings::miopenActivationMode_t_miopenActivationPASTHRU;
pub use bindings::miopenActivationMode_t_miopenActivationPOWER;
pub use bindings::miopenActivationMode_t_miopenActivationRELU;
pub use bindings::miopenActivationMode_t_miopenActivationSOFTRELU;
pub use bindings::miopenActivationMode_t_miopenActivationTANH;

// Softmax operations
pub use bindings::miopenCreateSoftmaxDescriptor;
pub use bindings::miopenGetSoftmaxDescriptor;
pub use bindings::miopenSetSoftmaxDescriptor;
pub use bindings::miopenSoftmaxBackward;
pub use bindings::miopenSoftmaxBackward_V2;
pub use bindings::miopenSoftmaxDescriptor_t;
pub use bindings::miopenSoftmaxForward;
pub use bindings::miopenSoftmaxForward_V2;

// Softmax enums
pub use bindings::miopenSoftmaxAlgorithm_t;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_ACCURATE;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_FAST;
pub use bindings::miopenSoftmaxAlgorithm_t_MIOPEN_SOFTMAX_LOG;

pub use bindings::miopenSoftmaxMode_t;
pub use bindings::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_CHANNEL;
pub use bindings::miopenSoftmaxMode_t_MIOPEN_SOFTMAX_MODE_INSTANCE;

// Fusion operations
pub use bindings::miopenCompileFusionPlan;
pub use bindings::miopenCreateFusionPlan;
pub use bindings::miopenCreateOpActivationBackward;
pub use bindings::miopenCreateOpActivationForward;
pub use bindings::miopenCreateOpBatchNormBackward;
pub use bindings::miopenCreateOpBatchNormForward;
pub use bindings::miopenCreateOpBatchNormInference;
pub use bindings::miopenCreateOpBiasForward;
pub use bindings::miopenCreateOpConvForward;
pub use bindings::miopenCreateOperatorArgs;
pub use bindings::miopenDestroyFusionPlan;
pub use bindings::miopenDestroyOperatorArgs;
pub use bindings::miopenExecuteFusionPlan;
pub use bindings::miopenFusionOpDescriptor_t;
pub use bindings::miopenFusionPlanConvolutionGetAlgo;
pub use bindings::miopenFusionPlanConvolutionSetAlgo;
pub use bindings::miopenFusionPlanDescriptor_t;
pub use bindings::miopenFusionPlanGetOp;
pub use bindings::miopenFusionPlanGetWorkSpaceSize;
pub use bindings::miopenOperatorArgs_t;
pub use bindings::miopenSetOpArgsActivBackward;
pub use bindings::miopenSetOpArgsActivForward;
pub use bindings::miopenSetOpArgsBatchNormBackward;
pub use bindings::miopenSetOpArgsBatchNormForward;
pub use bindings::miopenSetOpArgsBatchNormInference;
pub use bindings::miopenSetOpArgsBiasForward;
pub use bindings::miopenSetOpArgsConvForward;

// Fusion enums
pub use bindings::miopenFusionDirection_t;
pub use bindings::miopenFusionDirection_t_miopenHorizontalFusion;
pub use bindings::miopenFusionDirection_t_miopenVerticalFusion;

// RNN operations
pub use bindings::miopenAllocatorFunction;
pub use bindings::miopenCreateRNNDescriptor;
pub use bindings::miopenDeallocatorFunction;
pub use bindings::miopenDestroyRNNDescriptor;
pub use bindings::miopenGetRNNDescriptor;
pub use bindings::miopenGetRNNDescriptor_V2;
pub use bindings::miopenGetRNNHiddenTensorSize;
pub use bindings::miopenGetRNNInputTensorSize;
pub use bindings::miopenGetRNNLayerBias;
pub use bindings::miopenGetRNNLayerBiasOffset;
pub use bindings::miopenGetRNNLayerBiasSize;
pub use bindings::miopenGetRNNLayerParam;
pub use bindings::miopenGetRNNLayerParamOffset;
pub use bindings::miopenGetRNNLayerParamSize;
pub use bindings::miopenGetRNNParamsSize;
pub use bindings::miopenGetRNNTempSpaceSizes;
pub use bindings::miopenGetRNNTrainingReserveSize;
pub use bindings::miopenGetRNNWorkspaceSize;
pub use bindings::miopenRNNBackwardData;
pub use bindings::miopenRNNBackwardWeights;
pub use bindings::miopenRNNBaseLayout_t;
pub use bindings::miopenRNNDescriptor_t;
pub use bindings::miopenRNNForward;
pub use bindings::miopenRNNForwardInference;
pub use bindings::miopenRNNForwardTraining;
pub use bindings::miopenSetRNNDescriptor;
pub use bindings::miopenSetRNNDescriptor_V2;
pub use bindings::miopenSetRNNLayerBias;
pub use bindings::miopenSetRNNLayerParam;

// RNN enums
pub use bindings::miopenRNNMode_t;
pub use bindings::miopenRNNMode_t_miopenGRU;
pub use bindings::miopenRNNMode_t_miopenLSTM;
pub use bindings::miopenRNNMode_t_miopenRNNRELU;
pub use bindings::miopenRNNMode_t_miopenRNNTANH;

pub use bindings::miopenRNNInputMode_t;
pub use bindings::miopenRNNInputMode_t_miopenRNNlinear;
pub use bindings::miopenRNNInputMode_t_miopenRNNskip;

pub use bindings::miopenRNNAlgo_t;
pub use bindings::miopenRNNAlgo_t_miopenRNNdefault;
pub use bindings::miopenRNNAlgo_t_miopenRNNfundamental;

pub use bindings::miopenRNNDirectionMode_t;
pub use bindings::miopenRNNDirectionMode_t_miopenRNNbidirection;
pub use bindings::miopenRNNDirectionMode_t_miopenRNNunidirection;

pub use bindings::miopenRNNBiasMode_t;
pub use bindings::miopenRNNBiasMode_t_miopenRNNNoBias;
pub use bindings::miopenRNNBiasMode_t_miopenRNNwithBias;

pub use bindings::miopenRNNPaddingMode_t;
pub use bindings::miopenRNNPaddingMode_t_miopenRNNIONotPadded;
pub use bindings::miopenRNNPaddingMode_t_miopenRNNIOWithPadding;

pub use bindings::miopenRNNFWDMode_t;
pub use bindings::miopenRNNFWDMode_t_miopenRNNInference;
pub use bindings::miopenRNNFWDMode_t_miopenRNNTraining;

// Dropout operations
pub use bindings::miopenCreateDropoutDescriptor;
pub use bindings::miopenDestroyDropoutDescriptor;
pub use bindings::miopenDropoutBackward;
pub use bindings::miopenDropoutDescriptor_t;
pub use bindings::miopenDropoutForward;
pub use bindings::miopenDropoutGetReserveSpaceSize;
pub use bindings::miopenDropoutGetStatesSize;
pub use bindings::miopenGetDropoutDescriptor;
pub use bindings::miopenRestoreDropoutDescriptor;
pub use bindings::miopenSetDropoutDescriptor;

// Dropout enums
pub use bindings::miopenRNGType_t;
pub use bindings::miopenRNGType_t_MIOPEN_RNG_PSEUDO_XORWOW;

// Tensor reduction operations
pub use bindings::miopenCreateReduceTensorDescriptor;
pub use bindings::miopenDestroyReduceTensorDescriptor;
pub use bindings::miopenGetReduceTensorDescriptor;
pub use bindings::miopenGetReductionIndicesSize;
pub use bindings::miopenGetReductionWorkspaceSize;
pub use bindings::miopenReduceTensor;
pub use bindings::miopenReduceTensorDescriptor_t;
pub use bindings::miopenSetReduceTensorDescriptor;

// Tensor reduction enums
pub use bindings::miopenReduceTensorOp_t;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_ADD;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AMAX;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_AVG;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MAX;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MIN;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_MUL;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM1;
pub use bindings::miopenReduceTensorOp_t_MIOPEN_REDUCE_TENSOR_NORM2;

pub use bindings::miopenNanPropagation_t;
pub use bindings::miopenNanPropagation_t_MIOPEN_NOT_PROPAGATE_NAN;
pub use bindings::miopenNanPropagation_t_MIOPEN_PROPAGATE_NAN;

pub use bindings::miopenReduceTensorIndices_t;
pub use bindings::miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES;
pub use bindings::miopenReduceTensorIndices_t_MIOPEN_REDUCE_TENSOR_NO_INDICES;

pub use bindings::miopenCTCLoss;
pub use bindings::miopenCTCLossAlgo_t;
pub use bindings::miopenCTCLossDescriptor_t;
pub use bindings::miopenConvAlgorithm_t;
pub use bindings::miopenCreateCTCLossDescriptor;
pub use bindings::miopenCreateMhaDescriptor;
pub use bindings::miopenDestroyCTCLossDescriptor;
pub use bindings::miopenGetCTCLossDescriptor;
pub use bindings::miopenGetCTCLossWorkspaceSize;
pub use bindings::miopenGetConvolutionSpatialDim;
pub use bindings::miopenGetMhaDescriptor;
pub use bindings::miopenGetRNNPaddingMode;
pub use bindings::miopenGetSolutionSolverId;
pub use bindings::miopenIndicesType_t;
pub use bindings::miopenIndicesType_t_MIOPEN_8BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_16BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_32BIT_INDICES;
pub use bindings::miopenIndicesType_t_MIOPEN_64BIT_INDICES;
pub use bindings::miopenMhaDescriptor_t;
pub use bindings::miopenMhaMask_t;
pub use bindings::miopenMhaMask_t_miopenMhaMaskCausal;
pub use bindings::miopenMhaMask_t_miopenMhaMaskNone;
pub use bindings::miopenRNNBackwardSeqData;
pub use bindings::miopenRNNBackwardWeightsSeqTensor;
pub use bindings::miopenRNNBaseLayout_t_miopenRNNDataUnknownLayout;
pub use bindings::miopenRunSolution;
pub use bindings::miopenSetCTCLossDescriptor;
pub use bindings::miopenSetMhaDescriptor;
pub use bindings::miopenSetNdTensorDescriptorWithLayout;
pub use bindings::miopenSetRNNPaddingMode;
pub use bindings::miopenSetTransposeConvNdOutputPadding;
pub use bindings::miopenSetTransposeConvOutputPadding;
pub use bindings::miopenSolution_t;
pub use bindings::miopenTensorArgument_t;
pub use bindings::miopenTensorArgumentId_t;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxDK;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxDQ;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxDS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxDV;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaAmaxS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaBias;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDK;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDQ;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDV;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleDO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleDS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleK;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleQ;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDescaleV;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDropoutOffset;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDropoutProbability;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaDropoutSeed;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaK;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaM;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaMask;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaQ;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleDK;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleDQ;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleDS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleDV;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleO;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaScaleS;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaV;
pub use bindings::miopenTensorArgumentId_t_miopenTensorMhaZInv;
// Other needed functions and types
// Add more as needed...
