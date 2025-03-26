// include/rocwmma.h
// Wrapper include file for ROCWmma bindings with workarounds for AMD intrinsics

#ifndef ROCWMMA_BINDINGS_WRAPPER
#define ROCWMMA_BINDINGS_WRAPPER

// Define missing AMD GPU intrinsics for bindgen
#define __builtin_amdgcn_ds_bpermute(a, b) (0)
#define __builtin_amdgcn_ds_permute(a, b) (0)
#define __builtin_amdgcn_mov_dpp(a, b, c, d, e) (0)
#define __builtin_amdgcn_uicmp(a, b, c) (0)
#define __builtin_amdgcn_mbcnt_lo(a, b) (0)
#define __builtin_amdgcn_mbcnt_hi(a, b) (0)

// Define missing architecture-specific macros
#define __AMDGCN_WAVEFRONT_SIZE 64

// ADDED: Define __hip_internal as an empty namespace BEFORE including headers
// This might satisfy the parser when it encounters __hip_internal::something<T>
namespace __hip_internal {}

// Headers added via --include in build.rs (string, type_traits) are processed first by clang

// Now include the actual rocwmma headers
#include <hip/hip_runtime_api.h>
#include <rocwmma/rocwmma-version.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>

#endif // ROCWMMA_BINDINGS_WRAPPER