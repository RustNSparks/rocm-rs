// src/rocblas/mod.rs

// Private modules
pub mod error;
pub mod handle;
pub mod level1;
pub mod level2;
pub mod level3;
pub mod types;
pub mod utils;

// We need to make this public for the rest of the crate
// but don't necessarily want to expose it to users
#[allow(warnings)]
pub(crate) mod bindings;

// Public re-export of FFI for internal use
mod async_ops;
pub mod ffi;

// Re-export the main components for the public API
pub use error::{Error, Result};
pub use handle::Handle;
pub use level1::{
    amax,
    amax_batched,
    amax_strided_batched,
    amin,
    amin_batched,
    amin_strided_batched,
    asum,
    // batched variants
    asum_batched,
    // strided batched variants
    asum_strided_batched,
    axpy,
    axpy_batched,
    axpy_strided_batched,
    copy,
    copy_batched,
    copy_strided_batched,
    dot,
    dot_batched,
    dot_strided_batched,
    dotc,
    dotc_batched,
    dotc_strided_batched,
    dotu,
    dotu_batched,
    dotu_strided_batched,
    nrm2,
    nrm2_batched,
    nrm2_strided_batched,
    rot,
    rot_batched,
    rot_strided_batched,
    rotg,
    rotg_batched,
    rotg_strided_batched,
    rotm,
    rotm_batched,
    rotm_strided_batched,
    rotmg,
    rotmg_batched,
    rotmg_strided_batched,
    scal,
    scal_batched,
    scal_strided_batched,
    swap,
    swap_batched,
    swap_strided_batched,
};
pub use level2::{
    gbmv,
    // batched variants
    gbmv_batched,
    // strided batched variants
    gbmv_strided_batched,
    gemv,
    gemv_batched,
    gemv_strided_batched,
    hbmv,
    hbmv_batched,
    hbmv_strided_batched,
};
pub use level3::{gemm, gemm_batched, gemm_strided_batched};
pub use types::{
    rocblas_bfloat16, rocblas_datatype, rocblas_diagonal, rocblas_double_complex, rocblas_fill,
    rocblas_float_complex, rocblas_half, rocblas_operation, rocblas_side,
};
pub use utils::{
    AtomicsMode, GemmAlgo, GemmFlags, LayerMode, MathMode, PerformanceMetric, PointerMode,
    get_atomics_mode, get_math_mode, get_performance_metric, get_pointer_mode, set_atomics_mode,
    set_math_mode, set_performance_metric, set_pointer_mode,
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
