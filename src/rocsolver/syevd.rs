// src/rocsolver/syevd.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::Fill;
use crate::rocsolver::types::Evect;

/// Computes all eigenvalues and, optionally, eigenvectors of a symmetric matrix
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `evect` - Whether to compute eigenvectors or eigenvalues only
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `W` - Array for eigenvalues
/// * `info` - Success or failure indicator
pub fn syevd_float(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: &mut [f32],
    lda: i32,
    W: &mut [f32],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_ssyevd(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            W.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes all eigenvalues and, optionally, eigenvectors of a symmetric matrix (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `evect` - Whether to compute eigenvectors or eigenvalues only
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `W` - Array for eigenvalues
/// * `info` - Success or failure indicator
pub fn syevd_double(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: &mut [f64],
    lda: i32,
    W: &mut [f64],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dsyevd(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            W.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes all eigenvalues and, optionally, eigenvectors of a hermitian matrix (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `evect` - Whether to compute eigenvectors or eigenvalues only
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `W` - Array for eigenvalues
/// * `info` - Success or failure indicator
pub fn heevd_complex_float(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    W: &mut [f32],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cheevd(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            W.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes all eigenvalues and, optionally, eigenvectors of a hermitian matrix (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `evect` - Whether to compute eigenvectors or eigenvalues only
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `W` - Array for eigenvalues
/// * `info` - Success or failure indicator
pub fn heevd_complex_double(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    W: &mut [f64],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zheevd(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            W.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}