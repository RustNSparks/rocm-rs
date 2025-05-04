// src/rocsolver/larft.rs

use crate::rocblas::handle::Handle;
use crate::rocblas::ffi::{rocblas_float_complex, rocblas_double_complex};
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::ffi;
use crate::rocsolver::types::{Direct, Storev};

/// Generates the triangular factor T of a block reflector H
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `direct` - Specifies the direction for Householder matrices application
/// * `storev` - Specifies how the Householder vectors are stored in matrix V
/// * `n` - Order (size) of the block reflector
/// * `k` - Number of Householder matrices forming H
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `tau` - Vector of all Householder scalars
/// * `T` - Output triangular factor
/// * `ldt` - Leading dimension of T
pub fn larft_float(
    handle: &Handle,
    direct: Direct,
    storev: Storev,
    n: i32,
    k: i32,
    V: &mut [f32],
    ldv: i32,
    tau: &mut [f32],
    T: &mut [f32],
    ldt: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_slarft(
            handle.as_raw(),
            direct.into(),
            storev.into(),
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            tau.as_mut_ptr(),
            T.as_mut_ptr(),
            ldt,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates the triangular factor T of a block reflector H (double precision)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `direct` - Specifies the direction for Householder matrices application
/// * `storev` - Specifies how the Householder vectors are stored in matrix V
/// * `n` - Order (size) of the block reflector
/// * `k` - Number of Householder matrices forming H
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `tau` - Vector of all Householder scalars
/// * `T` - Output triangular factor
/// * `ldt` - Leading dimension of T
pub fn larft_double(
    handle: &Handle,
    direct: Direct,
    storev: Storev,
    n: i32,
    k: i32,
    V: &mut [f64],
    ldv: i32,
    tau: &mut [f64],
    T: &mut [f64],
    ldt: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_dlarft(
            handle.as_raw(),
            direct.into(),
            storev.into(),
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            tau.as_mut_ptr(),
            T.as_mut_ptr(),
            ldt,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates the triangular factor T of a block reflector H (complex)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `direct` - Specifies the direction for Householder matrices application
/// * `storev` - Specifies how the Householder vectors are stored in matrix V
/// * `n` - Order (size) of the block reflector
/// * `k` - Number of Householder matrices forming H
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `tau` - Vector of all Householder scalars
/// * `T` - Output triangular factor
/// * `ldt` - Leading dimension of T
pub fn larft_complex_float(
    handle: &Handle,
    direct: Direct,
    storev: Storev,
    n: i32,
    k: i32,
    V: &mut [rocblas_float_complex],
    ldv: i32,
    tau: &mut [rocblas_float_complex],
    T: &mut [rocblas_float_complex],
    ldt: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_clarft(
            handle.as_raw(),
            direct.into(),
            storev.into(),
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            tau.as_mut_ptr(),
            T.as_mut_ptr(),
            ldt,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}

/// Generates the triangular factor T of a block reflector H (complex double)
///
/// # Arguments
/// * `handle` - RocBLAS handle
/// * `direct` - Specifies the direction for Householder matrices application
/// * `storev` - Specifies how the Householder vectors are stored in matrix V
/// * `n` - Order (size) of the block reflector
/// * `k` - Number of Householder matrices forming H
/// * `V` - Matrix of Householder vectors
/// * `ldv` - Leading dimension of V
/// * `tau` - Vector of all Householder scalars
/// * `T` - Output triangular factor
/// * `ldt` - Leading dimension of T
pub fn larft_complex_double(
    handle: &Handle,
    direct: Direct,
    storev: Storev,
    n: i32,
    k: i32,
    V: &mut [rocblas_double_complex],
    ldv: i32,
    tau: &mut [rocblas_double_complex],
    T: &mut [rocblas_double_complex],
    ldt: i32,
) -> Result<()> {
    unsafe {
        let status = ffi::rocsolver_zlarft(
            handle.as_raw(),
            direct.into(),
            storev.into(),
            n,
            k,
            V.as_mut_ptr(),
            ldv,
            tau.as_mut_ptr(),
            T.as_mut_ptr(),
            ldt,
        );
        if status != ffi::rocblas_status__rocblas_status_success {
            return Err(Error::new(status));
        }
        Ok(())
    }
}