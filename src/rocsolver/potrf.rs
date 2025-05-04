// src/rocsolver/potrf.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::Fill;

// continuing src/rocsolver/potrf.rs

/// Computes the Cholesky factorization of a symmetric positive definite matrix
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `info` - Success or failure indicator
pub fn potrf_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f32],
    lda: i32,
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_spotrf(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a symmetric positive definite matrix (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `info` - Success or failure indicator
pub fn potrf_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f64],
    lda: i32,
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dpotrf(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a hermitian positive definite matrix (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `info` - Success or failure indicator
pub fn potrf_complex_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cpotrf(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a hermitian positive definite matrix (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `info` - Success or failure indicator
pub fn potrf_complex_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zpotrf(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of symmetric positive definite matrices (batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_batched_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &[*mut f32],
    lda: i32,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_spotrf_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_ptr(),
            lda,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of symmetric positive definite matrices (double precision, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_batched_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &[*mut f64],
    lda: i32,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dpotrf_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_ptr(),
            lda,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of hermitian positive definite matrices (complex, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_batched_complex_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &[*mut rocblas_float_complex],
    lda: i32,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cpotrf_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_ptr(),
            lda,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of hermitian positive definite matrices (complex double, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_batched_complex_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &[*mut rocblas_double_complex],
    lda: i32,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zpotrf_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_ptr(),
            lda,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of symmetric positive definite matrices (strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_strided_batched_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f32],
    lda: i32,
    strideA: i64,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_spotrf_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of symmetric positive definite matrices (double precision, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_strided_batched_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [f64],
    lda: i32,
    strideA: i64,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dpotrf_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of hermitian positive definite matrices (complex, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_strided_batched_complex_float(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    strideA: i64,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cpotrf_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the Cholesky factorization of a batch of hermitian positive definite matrices (complex double, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `uplo` - Specifies whether the upper or lower triangular part is stored
/// * `n` - Order of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `info` - Array of success or failure indicators
/// * `batch_count` - Number of matrices in the batch
pub fn potrf_strided_batched_complex_double(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    strideA: i64,
    info: &mut [i32],
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zpotrf_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            info.as_mut_ptr(),
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}