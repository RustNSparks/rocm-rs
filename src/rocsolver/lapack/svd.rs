// src/rocsolver/lapack/svd.rs
//! Singular Value Decomposition (SVD) operations.
//!
//! This module provides safe wrappers for SVD computations:
//!
//! - [`gesvd`] - Full SVD computation
//!
//! Note: Batched variants are not yet implemented due to complex stride requirements.

use crate::rocblas::Handle;
use crate::rocblas::ffi as rocblas_ffi;
use crate::rocsolver::bindings;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::types::{Complex32, Complex64, Svect, Workmode};

// Type alias for handle - we use rocblas handle but need to cast for rocsolver bindings
type RocblasHandle = rocblas_ffi::rocblas_handle;
type RocblasStatus = rocblas_ffi::rocblas_status;

/// Cast rocblas handle to rocsolver bindings handle type.
#[inline]
fn cast_handle(handle: RocblasHandle) -> bindings::rocblas_handle {
    handle as bindings::rocblas_handle
}

// ============================================================================
// Type traits for generic implementations
// ============================================================================

/// Trait for types that support SVD (gesvd).
pub trait GesvdType: Sized + Copy {
    /// The real type for singular values.
    type RealType: Copy;

    /// Compute the SVD of a general matrix.
    unsafe fn gesvd(
        handle: RocblasHandle,
        left_svect: bindings::rocblas_svect,
        right_svect: bindings::rocblas_svect,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        S: *mut Self::RealType,
        U: *mut Self,
        ldu: i32,
        V: *mut Self,
        ldv: i32,
        E: *mut Self::RealType,
        fast_alg: bindings::rocblas_workmode,
        info: *mut i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Trait implementations for f32
// ============================================================================

impl GesvdType for f32 {
    type RealType = f32;

    unsafe fn gesvd(
        handle: RocblasHandle,
        left_svect: bindings::rocblas_svect,
        right_svect: bindings::rocblas_svect,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        S: *mut Self::RealType,
        U: *mut Self,
        ldu: i32,
        V: *mut Self,
        ldv: i32,
        E: *mut Self::RealType,
        fast_alg: bindings::rocblas_workmode,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgesvd(
            cast_handle(handle),
            left_svect,
            right_svect,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            E,
            fast_alg,
            info,
        )
    }
}

// ============================================================================
// Trait implementations for f64
// ============================================================================

impl GesvdType for f64 {
    type RealType = f64;

    unsafe fn gesvd(
        handle: RocblasHandle,
        left_svect: bindings::rocblas_svect,
        right_svect: bindings::rocblas_svect,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        S: *mut Self::RealType,
        U: *mut Self,
        ldu: i32,
        V: *mut Self,
        ldv: i32,
        E: *mut Self::RealType,
        fast_alg: bindings::rocblas_workmode,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgesvd(
            cast_handle(handle),
            left_svect,
            right_svect,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            E,
            fast_alg,
            info,
        )
    }
}

// ============================================================================
// Trait implementations for Complex32
// ============================================================================

impl GesvdType for Complex32 {
    type RealType = f32;

    unsafe fn gesvd(
        handle: RocblasHandle,
        left_svect: bindings::rocblas_svect,
        right_svect: bindings::rocblas_svect,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        S: *mut Self::RealType,
        U: *mut Self,
        ldu: i32,
        V: *mut Self,
        ldv: i32,
        E: *mut Self::RealType,
        fast_alg: bindings::rocblas_workmode,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgesvd(
            cast_handle(handle),
            left_svect,
            right_svect,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            E,
            fast_alg,
            info,
        )
    }
}

// ============================================================================
// Trait implementations for Complex64
// ============================================================================

impl GesvdType for Complex64 {
    type RealType = f64;

    unsafe fn gesvd(
        handle: RocblasHandle,
        left_svect: bindings::rocblas_svect,
        right_svect: bindings::rocblas_svect,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        S: *mut Self::RealType,
        U: *mut Self,
        ldu: i32,
        V: *mut Self,
        ldv: i32,
        E: *mut Self::RealType,
        fast_alg: bindings::rocblas_workmode,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgesvd(
            cast_handle(handle),
            left_svect,
            right_svect,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            E,
            fast_alg,
            info,
        )
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Computes the Singular Value Decomposition (SVD) of a general m-by-n matrix A.
///
/// The SVD is written as:
///   A = U * S * V^T  (for real matrices)
///   A = U * S * V^H  (for complex matrices)
///
/// where S is an m-by-n matrix with real non-negative diagonal elements (the singular values),
/// U is an m-by-m orthogonal/unitary matrix, and V is an n-by-n orthogonal/unitary matrix.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `left_svect` - Specifies how to compute U (All, Singular, Overwrite, None)
/// * `right_svect` - Specifies how to compute V (All, Singular, Overwrite, None)
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `A` - Device pointer to m-by-n matrix (overwritten on output)
/// * `lda` - Leading dimension of A (>= max(1,m))
/// * `S` - Device pointer to min(m,n) singular values in decreasing order
/// * `U` - Device pointer to m-by-m (or m-by-min(m,n)) matrix U
/// * `ldu` - Leading dimension of U
/// * `V` - Device pointer to n-by-n (or min(m,n)-by-n) matrix V
/// * `ldv` - Leading dimension of V
/// * `E` - Device pointer to min(m,n)-1 superdiagonal elements of intermediate bidiagonal
/// * `fast_alg` - Workspace mode (InPlace or OutOfPlace)
/// * `info` - Device pointer to convergence info (0 = success, >0 = did not converge)
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
///
/// # Example
/// ```rust,no_run
/// use rocm_rs::{rocblas::Handle, rocsolver::{self, Svect, Workmode}};
///
/// let handle = Handle::new().unwrap();
/// let m = 4i32;
/// let n = 3i32;
/// // Allocate device memory for A, S, U, V, E, info...
/// // rocsolver::gesvd(&handle, Svect::All, Svect::All, m, n, A, m, S, U, m, V, n, E, Workmode::OutOfPlace, info).unwrap();
/// ```
#[inline]
pub fn gesvd<T: GesvdType>(
    handle: &Handle,
    left_svect: Svect,
    right_svect: Svect,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    S: *mut T::RealType,
    U: *mut T,
    ldu: i32,
    V: *mut T,
    ldv: i32,
    E: *mut T::RealType,
    fast_alg: Workmode,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe {
        T::gesvd(
            handle.as_raw(),
            left_svect.into(),
            right_svect.into(),
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            E,
            fast_alg.into(),
            info,
        )
    };
    Error::from_status(status)
}
