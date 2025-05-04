// Update to src/rocsolver/mod.rs

// Private modules
pub mod error;
pub mod handle;
pub mod types;
pub mod bindings;

// Public re-export of FFI for internal use
pub mod ffi;

// Re-export the main components for the public API
pub use error::{Error, Result};
pub use handle::Handle;
pub use types::{
    Direct, Evect, Eform, Eorder, Erange, Esort, Storev, Svect, Workmode,
    RfinfoMode, RfInfo,
};

// Re-export from rocBLAS types that RocSOLVER uses
pub use crate::rocblas::{
    PointerMode, 
    rocblas_float_complex, rocblas_double_complex, rocblas_half,
};

// Utility functions
pub use utils::{
    get_version_string, log_begin, log_end, log_set_layer_mode,
    log_set_max_levels, log_restore_defaults, log_write_profile,
    log_flush_profile,
};

// Re-export implementation modules
pub use lacgv::*;
pub use laswp::*;
// Continuing the updates to src/rocsolver/mod.rs

// Re-export implementation modules
pub use lacgv::*;
pub use laswp::*;
pub use larfg::*;
pub use larft::*;
pub use larf::*;
pub use larfb::*;
// pub use labrd::*;
// pub use latrd::*;
// pub use lasyf::*;
pub use lauum::*;
pub use org2r::*;
pub use gebrd::*;
pub use sytrd::*;
pub use potrf::*;
pub use gesvd::*;
pub use syevd::*;

// Helper modules for implementation
mod utils;
mod lacgv;
mod laswp;
mod larfg;
mod larft;
mod larf;
mod larfb;
// mod labrd;
// mod latrd;
// mod lasyf;
mod lauum;
mod org2r;
mod gebrd;
mod sytrd;
mod potrf;
mod gesvd;
mod syevd;

// Since RocSOLVER uses rocBLAS handles, we can just re-use the handle creation
// functions from rocBLAS
pub use crate::rocblas::create_handle;