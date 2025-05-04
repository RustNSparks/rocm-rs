// src/rocsolver/sytrd.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::Fill;

/// Reduces a symmetric matrix to tridiagonal form
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of tridiagonal matrix
/// * `E` - Array for off-diagonal elements of tridiagonal matrix
/// * `tau` - Array for Householder scalars
pub fn sytrd_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f32],
    lda: i32,
    D: &mut [f32],
    E: &mut [f32],
    tau: &mut [f32],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_ssytrd(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tau.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Reduces a symmetric matrix to tridiagonal form (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of tridiagonal matrix
/// * `E` - Array for off-diagonal elements of tridiagonal matrix
/// * `tau` - Array for Householder scalars
pub fn sytrd_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f64],
    lda: i32,
    D: &mut [f64],
    E: &mut [f64],
    tau: &mut [f64],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dsytrd(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tau.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Reduces a hermitian matrix to tridiagonal form (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of tridiagonal matrix
/// * `E` - Array for off-diagonal elements of tridiagonal matrix
/// * `tau` - Array for Householder scalars
pub fn hetrd_complex_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    D: &mut [f32],
    E: &mut [f32],
    tau: &mut [rocblas_float_complex],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_chetrd(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tau.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Reduces a hermitian matrix to tridiagonal form (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of tridiagonal matrix
/// * `E` - Array for off-diagonal elements of tridiagonal matrix
/// * `tau` - Array for Householder scalars
pub fn hetrd_complex_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    D: &mut [f64],
    E: &mut [f64],
    tau: &mut [rocblas_double_complex],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zhetrd(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tau.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}