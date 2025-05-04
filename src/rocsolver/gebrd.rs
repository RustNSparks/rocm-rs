// src/rocsolver/gebrd.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;

/// Computes the bidiagonal form of a general matrix A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of B
/// * `E` - Array for off-diagonal elements of B
/// * `tauq` - Array for Householder scalars associated with Q
/// * `taup` - Array for Householder scalars associated with P
pub fn gebrd_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [f32],
    lda: i32,
    D: &mut [f32],
    E: &mut [f32],
    tauq: &mut [f32],
    taup: &mut [f32],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_sgebrd(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tauq.as_mut_ptr(),
            taup.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a general matrix A (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of B
/// * `E` - Array for off-diagonal elements of B
/// * `tauq` - Array for Householder scalars associated with Q
/// * `taup` - Array for Householder scalars associated with P
pub fn gebrd_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [f64],
    lda: i32,
    D: &mut [f64],
    E: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dgebrd(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tauq.as_mut_ptr(),
            taup.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a general matrix A (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of B
/// * `E` - Array for off-diagonal elements of B
/// * `tauq` - Array for Householder scalars associated with Q
/// * `taup` - Array for Householder scalars associated with P
pub fn gebrd_complex_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    D: &mut [f32],
    E: &mut [f32],
    tauq: &mut [rocblas_float_complex],
    taup: &mut [rocblas_float_complex],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cgebrd(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tauq.as_mut_ptr(),
            taup.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a general matrix A (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of the matrix A
/// * `n` - Number of columns of the matrix A
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of A
/// * `D` - Array for diagonal elements of B
/// * `E` - Array for off-diagonal elements of B
/// * `tauq` - Array for Householder scalars associated with Q
/// * `taup` - Array for Householder scalars associated with P
pub fn gebrd_complex_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    D: &mut [f64],
    E: &mut [f64],
    tauq: &mut [rocblas_double_complex],
    taup: &mut [rocblas_double_complex],
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zgebrd(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            D.as_mut_ptr(),
            E.as_mut_ptr(),
            tauq.as_mut_ptr(),
            taup.as_mut_ptr(),
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// Additional functions for gebrd.rs

/// Computes the bidiagonal form of a batch of general matrices (batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_batched_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &[*mut f32],
    lda: i32,
    D: &mut [f32],
    strideD: i64,
    E: &mut [f32],
    strideE: i64,
    tauq: &mut [f32],
    strideQ: i64,
    taup: &mut [f32],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_sgebrd_batched(
            handle.as_raw(),
            m,
            n,
            A.as_ptr(),
            lda,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (double precision, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_batched_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &[*mut f64],
    lda: i32,
    D: &mut [f64],
    strideD: i64,
    E: &mut [f64],
    strideE: i64,
    tauq: &mut [f64],
    strideQ: i64,
    taup: &mut [f64],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dgebrd_batched(
            handle.as_raw(),
            m,
            n,
            A.as_ptr(),
            lda,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (complex, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_batched_complex_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &[*mut rocblas_float_complex],
    lda: i32,
    D: &mut [f32],
    strideD: i64,
    E: &mut [f32],
    strideE: i64,
    tauq: &mut [rocblas_float_complex],
    strideQ: i64,
    taup: &mut [rocblas_float_complex],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cgebrd_batched(
            handle.as_raw(),
            m,
            n,
            A.as_ptr(),
            lda,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (complex double, batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Array of matrices on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_batched_complex_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &[*mut rocblas_double_complex],
    lda: i32,
    D: &mut [f64],
    strideD: i64,
    E: &mut [f64],
    strideE: i64,
    tauq: &mut [rocblas_double_complex],
    strideQ: i64,
    taup: &mut [rocblas_double_complex],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zgebrd_batched(
            handle.as_raw(),
            m,
            n,
            A.as_ptr(),
            lda,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_strided_batched_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [f32],
    lda: i32,
    strideA: i64,
    D: &mut [f32],
    strideD: i64,
    E: &mut [f32],
    strideE: i64,
    tauq: &mut [f32],
    strideQ: i64,
    taup: &mut [f32],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_sgebrd_strided_batched(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (double precision, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_strided_batched_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [f64],
    lda: i32,
    strideA: i64,
    D: &mut [f64],
    strideD: i64,
    E: &mut [f64],
    strideE: i64,
    tauq: &mut [f64],
    strideQ: i64,
    taup: &mut [f64],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dgebrd_strided_batched(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (complex, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_strided_batched_complex_float(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
    strideA: i64,
    D: &mut [f32],
    strideD: i64,
    E: &mut [f32],
    strideE: i64,
    tauq: &mut [rocblas_float_complex],
    strideQ: i64,
    taup: &mut [rocblas_float_complex],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_cgebrd_strided_batched(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Computes the bidiagonal form of a batch of general matrices (complex double, strided batched)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `m` - Number of rows of each matrix
/// * `n` - Number of columns of each matrix
/// * `A` - Matrix on the GPU
/// * `lda` - Leading dimension of each matrix
/// * `strideA` - Stride between consecutive matrices
/// * `D` - Array for diagonal elements
/// * `strideD` - Stride between consecutive D arrays
/// * `E` - Array for off-diagonal elements
/// * `strideE` - Stride between consecutive E arrays
/// * `tauq` - Array for Householder scalars associated with Q
/// * `strideQ` - Stride between consecutive tauq arrays
/// * `taup` - Array for Householder scalars associated with P
/// * `strideP` - Stride between consecutive taup arrays
/// * `batch_count` - Number of matrices in the batch
pub fn gebrd_strided_batched_complex_double(
    handle: &Handle,
    m: i32,
    n: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
    strideA: i64,
    D: &mut [f64],
    strideD: i64,
    E: &mut [f64],
    strideE: i64,
    tauq: &mut [rocblas_double_complex],
    strideQ: i64,
    taup: &mut [rocblas_double_complex],
    strideP: i64,
    batch_count: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zgebrd_strided_batched(
            handle.as_raw(),
            m,
            n,
            A.as_mut_ptr(),
            lda,
            strideA,
            D.as_mut_ptr(),
            strideD,
            E.as_mut_ptr(),
            strideE,
            tauq.as_mut_ptr(),
            strideQ,
            taup.as_mut_ptr(),
            strideP,
            batch_count,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}