// src/rocsolver/laswp.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;

/// Performs a series of row interchanges on the matrix A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix to which row interchanges will be applied
/// * `lda` - Leading dimension of A
/// * `k1` - First index for which a row interchange will be done (1-based)
/// * `k2` - Last index for which a row interchange will be done (1-based)
/// * `ipiv` - Vector of pivot indices (1-based)
/// * `incx` - Stride between consecutive values of ipiv
pub fn laswp_float(
    handle: &Handle,
    n: i32,
    A: &mut [f32],
    lda: i32,
    k1: i32,
    k2: i32,
    ipiv: &[i32],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slaswp(
            handle.as_raw(),
            n,
            A.as_mut_ptr(),
            lda,
            k1,
            k2,
            ipiv.as_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Performs a series of row interchanges on the matrix A (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix to which row interchanges will be applied
/// * `lda` - Leading dimension of A
/// * `k1` - First index for which a row interchange will be done (1-based)
/// * `k2` - Last index for which a row interchange will be done (1-based)
/// * `ipiv` - Vector of pivot indices (1-based)
/// * `incx` - Stride between consecutive values of ipiv
pub fn laswp_double(
    handle: &Handle,
    n: i32,
    A: &mut [f64],
    lda: i32,
    k1: i32,
    k2: i32,
    ipiv: &[i32],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlaswp(
            handle.as_raw(),
            n,
            A.as_mut_ptr(),
            lda,
            k1,
            k2,
            ipiv.as_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Performs a series of row interchanges on the matrix A (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix to which row interchanges will be applied
/// * `lda` - Leading dimension of A
/// * `k1` - First index for which a row interchange will be done (1-based)
/// * `k2` - Last index for which a row interchange will be done (1-based)
/// * `ipiv` - Vector of pivot indices (1-based)
/// * `incx` - Stride between consecutive values of ipiv
pub fn laswp_complex_float(
    handle: &Handle,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    k1: i32,
    k2: i32,
    ipiv: &[i32],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_claswp(
            handle.as_raw(),
            n,
            A.as_mut_ptr(),
            lda,
            k1,
            k2,
            ipiv.as_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Performs a series of row interchanges on the matrix A (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix to which row interchanges will be applied
/// * `lda` - Leading dimension of A
/// * `k1` - First index for which a row interchange will be done (1-based)
/// * `k2` - Last index for which a row interchange will be done (1-based)
/// * `ipiv` - Vector of pivot indices (1-based)
/// * `incx` - Stride between consecutive values of ipiv
pub fn laswp_complex_double(
    handle: &Handle,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    k1: i32,
    k2: i32,
    ipiv: &[i32],
    incx: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlaswp(
            handle.as_raw(),
            n,
            A.as_mut_ptr(),
            lda,
            k1,
            k2,
            ipiv.as_ptr(),
            incx,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}