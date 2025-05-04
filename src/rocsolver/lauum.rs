// src/rocsolver/lauum.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::Fill;

/// Computes the product of the upper or lower triangular part of a matrix with its transpose
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part will be used
/// * `n` - Number of columns and rows of the matrix A
/// * `A` - Input/output matrix
/// * `lda` - Leading dimension of A
pub fn lauum_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f32],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slauum(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the product of the upper or lower triangular part of a matrix with its transpose (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part will be used
/// * `n` - Number of columns and rows of the matrix A
/// * `A` - Input/output matrix
/// * `lda` - Leading dimension of A
pub fn lauum_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f64],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlauum(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the product of the upper or lower triangular part of a matrix with its transpose (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part will be used
/// * `n` - Number of columns and rows of the matrix A
/// * `A` - Input/output matrix
/// * `lda` - Leading dimension of A
pub fn lauum_complex_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clauum(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the product of the upper or lower triangular part of a matrix with its transpose (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part will be used
/// * `n` - Number of columns and rows of the matrix A
/// * `A` - Input/output matrix
/// * `lda` - Leading dimension of A
pub fn lauum_complex_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlauum(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}