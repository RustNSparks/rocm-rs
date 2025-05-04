// src/rocsolver/larf.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocblas::types::Side;

/// Applies a Householder reflector H to a general matrix A
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_float(
    handle: &Handle,
    side: Side,
    m: i32,
    n: i32,
    x: &mut [f32],
    incx: i32,
    alpha: &f32,
    A: &mut [f32],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarf(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a Householder reflector H to a general matrix A (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_double(
    handle: &Handle,
    side: Side,
    m: i32,
    n: i32,
    x: &mut [f64],
    incx: i32,
    alpha: &f64,
    A: &mut [f64],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarf(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a Householder reflector H to a general matrix A (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_complex_float(
    handle: &Handle,
    side: Side,
    m: i32,
    n: i32,
    x: &mut [rocblas_float_complex],
    incx: i32,
    alpha: &rocblas_float_complex,
    A: &mut [rocblas_float_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarf(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a Householder reflector H to a general matrix A (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_complex_double(
    handle: &Handle,
    side: Side,
    m: i32,
    n: i32,
    x: &mut [rocblas_double_complex],
    incx: i32,
    alpha: &rocblas_double_complex,
    A: &mut [rocblas_double_complex],
    lda: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarf(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// 64-bit variants

/// Applies a Householder reflector H to a general matrix A (64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_float_64(
    handle: &Handle,
    side: Side,
    m: i64,
    n: i64,
    x: &mut [f32],
    incx: i64,
    alpha: &f32,
    A: &mut [f32],
    lda: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarf_64(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// src/rocsolver/larf.rs (continued)

/// Applies a Householder reflector H to a general matrix A (double precision, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_double_64(
    handle: &Handle,
    side: Side,
    m: i64,
    n: i64,
    x: &mut [f64],
    incx: i64,
    alpha: &f64,
    A: &mut [f64],
    lda: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarf_64(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a Householder reflector H to a general matrix A (complex, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_complex_float_64(
    handle: &Handle,
    side: Side,
    m: i64,
    n: i64,
    x: &mut [rocblas_float_complex],
    incx: i64,
    alpha: &rocblas_float_complex,
    A: &mut [rocblas_float_complex],
    lda: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarf_64(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Applies a Householder reflector H to a general matrix A (complex double, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `side` - Determines whether H is applied from the left or right
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `x` - Householder vector x
/// * `incx` - Stride between consecutive elements of x
/// * `alpha` - Householder scalar
/// * `A` - Input/output matrix A
/// * `lda` - Leading dimension of A
pub fn larf_complex_double_64(
    handle: &Handle,
    side: Side,
    m: i64,
    n: i64,
    x: &mut [rocblas_double_complex],
    incx: i64,
    alpha: &rocblas_double_complex,
    A: &mut [rocblas_double_complex],
    lda: i64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarf_64(
            handle.as_raw(),
            side.into(),
            m,
            n,
            x.as_mut_ptr(),
            incx,
            alpha,
            A.as_mut_ptr(),
            lda,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}