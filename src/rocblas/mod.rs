// src/rocblas/mod.rs

// Private modules
pub mod error;
pub mod handle;
pub mod types;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod utils;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
pub(crate) mod bindings;

// Public re-export of FFI for internal use
pub mod ffi;

// Re-export the main components for the public API
pub use error::{Error, Result};
pub use handle::Handle;
pub use types::{
    rocblas_float_complex, rocblas_double_complex, rocblas_half, rocblas_bfloat16,
    rocblas_operation, rocblas_fill, rocblas_diagonal, rocblas_side, rocblas_datatype,
};
pub use level1::{
    asum, axpy, copy, dot, dotu, dotc, nrm2, rot, rotg, rotm, rotmg, scal, swap,
    amax, amin,
    // batched variants
    asum_batched, axpy_batched, copy_batched, dot_batched, dotu_batched, dotc_batched,
    nrm2_batched, rot_batched, rotg_batched, rotm_batched, rotmg_batched, scal_batched,
    swap_batched, amax_batched, amin_batched,
    // strided batched variants
    asum_strided_batched, axpy_strided_batched, copy_strided_batched, dot_strided_batched,
    dotu_strided_batched, dotc_strided_batched, nrm2_strided_batched, rot_strided_batched,
    rotg_strided_batched, rotm_strided_batched, rotmg_strided_batched, scal_strided_batched,
    swap_strided_batched, amax_strided_batched, amin_strided_batched,
};
pub use level2::{
    gbmv, gemv, hbmv,
    // batched variants
    gbmv_batched, gemv_batched, hbmv_batched,
    // strided batched variants
    gbmv_strided_batched, gemv_strided_batched, hbmv_strided_batched,
};
pub use level3::{
    gemm, gemm_batched, gemm_strided_batched,
};
pub use utils::{
    PointerMode, AtomicsMode, PerformanceMetric, LayerMode, GemmAlgo, GemmFlags, MathMode,
    set_pointer_mode, get_pointer_mode,
    set_atomics_mode, get_atomics_mode,
    set_math_mode, get_math_mode,
    set_performance_metric, get_performance_metric,
};

/// Create a RocBLAS handle
pub fn create_handle() -> Result<Handle> {
    Handle::new()
}

/// Initialize RocBLAS
/// 
/// Note: In most cases, explicit initialization is not required
/// as handle creation will initialize the library
pub fn init() -> Result<()> {
    // Creating and immediately dropping a handle
    // will initialize rocBLAS and free resources
    let _ = create_handle()?;
    Ok(())
}