extern crate core;
pub mod error;
pub mod hip;
#[cfg(feature = "miopen")]
pub mod miopen;
pub mod rocblas;
pub mod rocfft;
pub mod rocrand;
#[cfg(feature = "rocsolver")]
pub mod rocsolver;

#[cfg(feature = "rocm_smi")]
pub mod rocmsmi;
// mod rocprofiler;
pub mod rocarray;
pub mod rocsparse;

#[cfg(feature = "macros")]
pub use rocm_kernel_macros;

// TEAM-507: bindgen_rocm for kernel compilation (mirrors bindgen_cuda)
pub mod bindgen_rocm;
