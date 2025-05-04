// src/rocsolver/gesvd.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocsolver::types::Svect;

/// Computes the singular value decomposition (SVD) of a general m-by-n matrix
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `leftv` - How to compute left singular vectors
/// * `rightv` - How to compute right singular vectors
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `S` - Array for singular values
/// * `U` - Matrix for left singular vectors
/// * `ldu` - Leading dimension of U
/// * `V` - Matrix for right singular vectors
/// * `ldv` - Leading dimension of V
/// * `E` - Superdiagonal elements of bidiagonal matrix (workspace)
/// * `info` - Success or failure indicator
pub fn gesvd_float(
    handle: &Handle,
    leftv: Svect,
    rightv: Svect,
    m: i32,
    n: i32,
    A: &mut [f32],
    lda: i32,
    S: &mut [f32],
    U: &mut [f32],
    ldu: i32,
    V: &mut [f32],
    ldv: i32,
    E: &mut [f32],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_sgesvd(
            handle.as_raw(),
            leftv.into(),
            rightv.into(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            S.as_mut_ptr(),
            U.as_mut_ptr(),
            ldu,
            V.as_mut_ptr(),
            ldv,
            E.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the singular value decomposition (SVD) of a general m-by-n matrix (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `leftv` - How to compute left singular vectors
/// * `rightv` - How to compute right singular vectors
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `S` - Array for singular values
/// * `U` - Matrix for left singular vectors
/// * `ldu` - Leading dimension of U
/// * `V` - Matrix for right singular vectors
/// * `ldv` - Leading dimension of V
/// * `E` - Superdiagonal elements of bidiagonal matrix (workspace)
/// * `info` - Success or failure indicator
pub fn gesvd_double(
    handle: &Handle,
    leftv: Svect,
    rightv: Svect,
    m: i32,
    n: i32,
    A: &mut [f64],
    lda: i32,
    S: &mut [f64],
    U: &mut [f64],
    ldu: i32,
    V: &mut [f64],
    ldv: i32,
    E: &mut [f64],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dgesvd(
            handle.as_raw(),
            leftv.into(),
            rightv.into(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            S.as_mut_ptr(),
            U.as_mut_ptr(),
            ldu,
            V.as_mut_ptr(),
            ldv,
            E.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the singular value decomposition (SVD) of a general m-by-n matrix (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `leftv` - How to compute left singular vectors
/// * `rightv` - How to compute right singular vectors
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `S` - Array for singular values
/// * `U` - Matrix for left singular vectors
/// * `ldu` - Leading dimension of U
/// * `V` - Matrix for right singular vectors
/// * `ldv` - Leading dimension of V
/// * `E` - Superdiagonal elements of bidiagonal matrix (workspace)
/// * `info` - Success or failure indicator
pub fn gesvd_complex_float(
    handle: &Handle,
    leftv: Svect,
    rightv: Svect,
    m: i32,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    S: &mut [f32],
    U: &mut [rocblas_float_complex],
    ldu: i32,
    V: &mut [rocblas_float_complex],
    ldv: i32,
    E: &mut [f32],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cgesvd(
            handle.as_raw(),
            leftv.into(),
            rightv.into(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            S.as_mut_ptr(),
            U.as_mut_ptr(),
            ldu,
            V.as_mut_ptr(),
            ldv,
            E.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the singular value decomposition (SVD) of a general m-by-n matrix (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `leftv` - How to compute left singular vectors
/// * `rightv` - How to compute right singular vectors
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `S` - Array for singular values
/// * `U` - Matrix for left singular vectors
/// * `ldu` - Leading dimension of U
/// * `V` - Matrix for right singular vectors
/// * `ldv` - Leading dimension of V
/// * `E` - Superdiagonal elements of bidiagonal matrix (workspace)
/// * `info` - Success or failure indicator
pub fn gesvd_complex_double(
    handle: &Handle,
    leftv: Svect,
    rightv: Svect,
    m: i32,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    S: &mut [f64],
    U: &mut [rocblas_double_complex],
    ldu: i32,
    V: &mut [rocblas_double_complex],
    ldv: i32,
    E: &mut [f64],
    info: &mut i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zgesvd(
            handle.as_raw(),
            leftv.into(),
            rightv.into(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            S.as_mut_ptr(),
            U.as_mut_ptr(),
            ldu,
            V.as_mut_ptr(),
            ldv,
            E.as_mut_ptr(),
            info as *mut _,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}