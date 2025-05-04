// src/hip/ffi.rs
//
// FFI bindings for the HIP API
// This file re-exports the necessary symbols from the auto-generated bindings

// We assume there's a bindings module that was auto-generated
// using bindgen or similar tool
use crate::hip::bindings;

// Re-export the necessary types, constants, and functions

// Error type and constants
pub use bindings::hipError_t;
pub use bindings::hipError_t_hipSuccess;
pub use bindings::hipError_t_hipErrorInvalidValue;
pub use bindings::hipError_t_hipErrorOutOfMemory;
pub use bindings::hipError_t_hipErrorMemoryAllocation;
pub use bindings::hipError_t_hipErrorNotInitialized;
pub use bindings::hipError_t_hipErrorInvalidDevice;
pub use bindings::hipError_t_hipErrorInvalidContext;
pub use bindings::hipError_t_hipErrorNotReady;

// Device handle and operations
pub use bindings::hipDevice_t;
pub use bindings::hipDeviceProp_tR0600;
pub use bindings::hipInit;
pub use bindings::hipGetDeviceCount;
pub use bindings::hipGetDevice;
pub use bindings::hipSetDevice;
pub use bindings::hipGetDevicePropertiesR0600;
pub use bindings::hipDriverGetVersion;
pub use bindings::hipRuntimeGetVersion;
pub use bindings::hipDeviceSynchronize;
pub use bindings::hipDeviceReset;
pub use bindings::hipGetLastError;
pub use bindings::hipGetErrorName;
pub use bindings::hipGetErrorString;

// Memory management
pub use bindings::hipMalloc;
pub use bindings::hipFree;
pub use bindings::hipHostMalloc;
pub use bindings::hipHostFree;
pub use bindings::hipMemcpy;
pub use bindings::hipMemcpyAsync;
pub use bindings::hipMemset;
pub use bindings::hipMemGetInfo;
pub use bindings::hipHostGetDevicePointer;

// Memory copy kinds
pub use bindings::hipMemcpyKind_hipMemcpyHostToHost;
pub use bindings::hipMemcpyKind_hipMemcpyHostToDevice;
pub use bindings::hipMemcpyKind_hipMemcpyDeviceToHost;
pub use bindings::hipMemcpyKind_hipMemcpyDeviceToDevice;
pub use bindings::hipMemcpyKind_hipMemcpyDefault;

// Host malloc flags
pub use bindings::hipHostMallocDefault;
pub use bindings::hipHostMallocPortable;
pub use bindings::hipHostMallocMapped;
pub use bindings::hipHostMallocWriteCombined;
pub use bindings::hipHostMallocNumaUser;
pub use bindings::hipHostMallocCoherent;
pub use bindings::hipHostMallocNonCoherent;

// Stream operations
pub use bindings::hipStream_t;
pub use bindings::hipStreamCreate;
pub use bindings::hipStreamCreateWithFlags;
pub use bindings::hipStreamCreateWithPriority;
pub use bindings::hipStreamDestroy;
pub use bindings::hipStreamSynchronize;
pub use bindings::hipStreamQuery;
pub use bindings::hipStreamWaitEvent;
pub use bindings::hipStreamAddCallback;
pub use bindings::hipStreamGetPriority;
pub use bindings::hipStreamGetFlags;
pub use bindings::hipStreamGetDevice;
pub use bindings::hipDeviceGetStreamPriorityRange;

// Event operations
pub use bindings::hipEvent_t;
pub use bindings::hipEventCreate;
pub use bindings::hipEventCreateWithFlags;
pub use bindings::hipEventDestroy;
pub use bindings::hipEventRecord;
pub use bindings::hipEventSynchronize;
pub use bindings::hipEventQuery;
pub use bindings::hipEventElapsedTime;

// Kernel launching
pub use bindings::dim3;
pub use bindings::hipModuleGetFunction;
pub use bindings::hipFunction_t;
pub use bindings::hipModuleLaunchKernel;
pub use bindings::hipLaunchKernel;

// Texture and surface references
pub use bindings::hipTextureObject_t;
pub use bindings::hipSurfaceObject_t;
pub use bindings::hipCreateTextureObject;
pub use bindings::hipDestroyTextureObject;
pub use bindings::hipCreateSurfaceObject;
pub use bindings::hipDestroySurfaceObject;

pub use bindings::hipModule_t;
pub use bindings::hipModuleUnload;
pub use bindings::hipModuleLoad;
pub use bindings::hipModuleLoadData;
pub use bindings::hipModuleLoadDataEx;
pub use bindings::hipJitOption;
pub use bindings::hipModuleGetGlobal;


// Other useful constants and types as needed for your implementation
// Add more imports as required by your wrapper implementation