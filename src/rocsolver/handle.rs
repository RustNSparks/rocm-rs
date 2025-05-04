// src/rocsolver/handle.rs

// RocSOLVER doesn't have its own handle type - it uses the rocBLAS handle
// So we'll just re-export the rocBLAS handle

pub use crate::rocblas::handle::Handle;