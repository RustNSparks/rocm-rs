// src/rocsolver/lacgv.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;

/// Conjugates the complex vector x
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in the vector x
/// * `x` - Vector to be conjugated
/// * `incx` - Stride between consecutive elements of x
pub fn lacgv(
    handle: &Handle,
    n: i32,
    x: &mut [rocblas_float_complex],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clacgv(
            handle.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Conjugates the complex vector x (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in the vector x
/// * `x` - Vector to be conjugated
/// * `incx` - Stride between consecutive elements of x
pub fn lacgv_double(
    handle: &Handle,
    n: i32,
    x: &mut [rocblas_double_complex],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlacgv(
            handle.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Conjugates the complex vector x (64-bit indices)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in the vector x
/// * `x` - Vector to be conjugated
/// * `incx` - Stride between consecutive elements of x
pub fn lacgv_64(
    handle: &Handle,
    n: i64,
    x: &mut [rocblas_float_complex],
    incx: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clacgv_64(
            handle.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Conjugates the complex vector x (double precision, 64-bit indices)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of elements in the vector x
/// * `x` - Vector to be conjugated
/// * `incx` - Stride between consecutive elements of x
pub fn lacgv_double_64(
    handle: &Handle,
    n: i64,
    x: &mut [rocblas_double_complex],
    incx: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlacgv_64(
            handle.as_raw(),
            n,
            x.as_mut_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}