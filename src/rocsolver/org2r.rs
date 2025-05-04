// src/rocsolver/org2r.rs

use crate::rocblas::handle::Handle;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;

/// Generates an m-by-n matrix Q with orthonormal columns
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix Q
/// * `n` - Number of columns of the matrix Q
/// * `k` - Number of Householder reflectors
/// * `A` - Input/output matrix, contains Householder vectors on entry and returns matrix Q
/// * `lda` - Leading dimension of A
/// * `ipiv` - Householder scalars
pub fn org2r_float(
    handle: &Handle,
    m: i32,
    n: i32,
    k: i32,
    A: &mut [f32],
    lda: i32,
    ipiv: &mut [f32],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_sorg2r(
            handle.as_raw(),
            m,
            n,
            k,
            A.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates an m-by-n matrix Q with orthonormal columns (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix Q
/// * `n` - Number of columns of the matrix Q
/// * `k` - Number of Householder reflectors
/// * `A` - Input/output matrix, contains Householder vectors on entry and returns matrix Q
/// * `lda` - Leading dimension of A
/// * `ipiv` - Householder scalars
pub fn org2r_double(
    handle: &Handle,
    m: i32,
    n: i32,
    k: i32,
    A: &mut [f64],
    lda: i32,
    ipiv: &mut [f64],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dorg2r(
            handle.as_raw(),
            m,
            n,
            k,
            A.as_mut_ptr(),
            lda,
            ipiv.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}