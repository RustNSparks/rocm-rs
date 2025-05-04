// src/rocsolver/larfb.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::{Side, Operation};
use crate::rocsolver::types::{Direct, Storev};

/// Applies a block reflector H to a general matrix A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Specifies from which side to apply H
/// * `trans` - Specifies whether the block reflector or its transpose/conjugate transpose is applied
/// * `direct` - Specifies the direction in which Householder matrices are applied
/// * `storev` - Specifies how Householder vectors are stored in matrix V
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `k` - Number of Householder matrices
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `T` - Triangular factor of the block reflector
/// * `ldt` - Leading dimension of T
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larfb_float(
    handle: &Handle,
    side: Side,
    trans: Operation,
    direct: Direct,
    storev: Storev,
    m: i32,
    n: i32,
    k: i32,
    V: &mut [f32],
    ldv: i32,
    T: &mut [f32],
    ldt: i32,
    A: &mut [f32],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarfb(
            handle.as_raw(),
            side.into(),
            trans.into(),
            direct.into(),
            storev.into(),
            m,
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            T.as_mut_ptr(),
            ldt,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a block reflector H to a general matrix A (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Specifies from which side to apply H
/// * `trans` - Specifies whether the block reflector or its transpose/conjugate transpose is applied
/// * `direct` - Specifies the direction in which Householder matrices are applied
/// * `storev` - Specifies how Householder vectors are stored in matrix V
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `k` - Number of Householder matrices
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `T` - Triangular factor of the block reflector
/// * `ldt` - Leading dimension of T
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larfb_double(
    handle: &Handle,
    side: Side,
    trans: Operation,
    direct: Direct,
    storev: Storev,
    m: i32,
    n: i32,
    k: i32,
    V: &mut [f64],
    ldv: i32,
    T: &mut [f64],
    ldt: i32,
    A: &mut [f64],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarfb(
            handle.as_raw(),
            side.into(),
            trans.into(),
            direct.into(),
            storev.into(),
            m,
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            T.as_mut_ptr(),
            ldt,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a block reflector H to a general matrix A (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Specifies from which side to apply H
/// * `trans` - Specifies whether the block reflector or its transpose/conjugate transpose is applied
/// * `direct` - Specifies the direction in which Householder matrices are applied
/// * `storev` - Specifies how Householder vectors are stored in matrix V
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `k` - Number of Householder matrices
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `T` - Triangular factor of the block reflector
/// * `ldt` - Leading dimension of T
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larfb_complex_float(
    handle: &Handle,
    side: Side,
    trans: Operation,
    direct: Direct,
    storev: Storev,
    m: i32,
    n: i32,
    k: i32,
    V: &mut [rocblas_float_complex],
    ldv: i32,
    T: &mut [rocblas_float_complex],
    ldt: i32,
    A: &mut [rocblas_float_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarfb(
            handle.as_raw(),
            side.into(),
            trans.into(),
            direct.into(),
            storev.into(),
            m,
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            T.as_mut_ptr(),
            ldt,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a block reflector H to a general matrix A (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Specifies from which side to apply H
/// * `trans` - Specifies whether the block reflector or its transpose/conjugate transpose is applied
/// * `direct` - Specifies the direction in which Householder matrices are applied
/// * `storev` - Specifies how Householder vectors are stored in matrix V
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `k` - Number of Householder matrices
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `T` - Triangular factor of the block reflector
/// * `ldt` - Leading dimension of T
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larfb_complex_double(
    handle: &Handle,
    side: Side,
    trans: Operation,
    direct: Direct,
    storev: Storev,
    m: i32,
    n: i32,
    k: i32,
    V: &mut [rocblas_double_complex],
    ldv: i32,
    T: &mut [rocblas_double_complex],
    ldt: i32,
    A: &mut [rocblas_double_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarfb(
            handle.as_raw(),
            side.into(),
            trans.into(),
            direct.into(),
            storev.into(),
            m,
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            T.as_mut_ptr(),
            ldt,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}