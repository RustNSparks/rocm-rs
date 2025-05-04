// src/rocsolver/larfg.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;

/// Generates a Householder reflector H of order n
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_float(
    handle: &Handle,
    n: i32,
    alpha: &mut f32,
    x: &mut [f32],
    incx: i32,
    tau: &mut f32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarfg(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_double(
    handle: &Handle,
    n: i32,
    alpha: &mut f64,
    x: &mut [f64],
    incx: i32,
    tau: &mut f64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarfg(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_complex_float(
    handle: &Handle,
    n: i32,
    alpha: &mut rocblas_float_complex,
    x: &mut [rocblas_float_complex],
    incx: i32,
    tau: &mut rocblas_float_complex,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarfg(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_complex_double(
    handle: &Handle,
    n: i32,
    alpha: &mut rocblas_double_complex,
    x: &mut [rocblas_double_complex],
    incx: i32,
    tau: &mut rocblas_double_complex,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarfg(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

// 64-bit versions

/// Generates a Householder reflector H of order n (64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_float_64(
    handle: &Handle,
    n: i64,
    alpha: &mut f32,
    x: &mut [f32],
    incx: i64,
    tau: &mut f32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarfg_64(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (double precision, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_double_64(
    handle: &Handle,
    n: i64,
    alpha: &mut f64,
    x: &mut [f64],
    incx: i64,
    tau: &mut f64,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarfg_64(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (complex, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_complex_float_64(
    handle: &Handle,
    n: i64,
    alpha: &mut rocblas_float_complex,
    x: &mut [rocblas_float_complex],
    incx: i64,
    tau: &mut rocblas_float_complex,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarfg_64(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates a Householder reflector H of order n (complex double, 64-bit)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `n` - Order (size) of reflector H
/// * `alpha` - Input scalar alpha, overwritten with beta on output
/// * `x` - Input vector x, overwritten with vector v on output
/// * `incx` - Stride between consecutive elements of x
/// * `tau` - Output scalar tau (Householder scalar)
pub fn larfg_complex_double_64(
    handle: &Handle,
    n: i64,
    alpha: &mut rocblas_double_complex,
    x: &mut [rocblas_double_complex],
    incx: i64,
    tau: &mut rocblas_double_complex,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarfg_64(
            handle.as_raw(),
            n,
            alpha,
            x.as_mut_ptr(),
            incx,
            tau,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}