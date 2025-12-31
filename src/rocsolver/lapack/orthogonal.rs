// src/rocsolver/lapack/orthogonal.rs
//! Orthogonal/Unitary matrix operations.
//!
//! This module provides safe wrappers for generating and applying orthogonal/unitary matrices:
//!
//! - [`orgqr`]/[`ungqr`] - Generate Q from QR factorization
//! - [`ormqr`]/[`unmqr`] - Apply Q from QR factorization
//!
//! Note: rocSOLVER does not provide batched variants for these operations.

use crate::rocblas::Handle;
use crate::rocblas::ffi as rocblas_ffi;
use crate::rocblas::types::{Operation, Side};
use crate::rocsolver::bindings;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::types::{Complex32, Complex64};

// Type alias for handle - we use rocblas handle but need to cast for rocsolver bindings
type RocblasHandle = rocblas_ffi::rocblas_handle;
type RocblasStatus = rocblas_ffi::rocblas_status;

/// Cast rocblas handle to rocsolver bindings handle type.
#[inline]
fn cast_handle(handle: RocblasHandle) -> bindings::rocblas_handle {
    handle as bindings::rocblas_handle
}

// ============================================================================
// Type traits for real orthogonal operations
// ============================================================================

/// Trait for real types that support orgqr (generate orthogonal Q from QR).
pub trait OrgqrType: Sized + Copy {
    /// Generate the orthogonal matrix Q from QR factorization.
    unsafe fn orgqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus;
}

/// Trait for real types that support ormqr (apply orthogonal Q from QR).
pub trait OrmqrType: Sized + Copy {
    /// Apply Q from QR factorization to a matrix C.
    unsafe fn ormqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Type traits for complex unitary operations
// ============================================================================

/// Trait for complex types that support ungqr (generate unitary Q from QR).
pub trait UngqrType: Sized + Copy {
    /// Generate the unitary matrix Q from QR factorization.
    unsafe fn ungqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus;
}

/// Trait for complex types that support unmqr (apply unitary Q from QR).
pub trait UnmqrType: Sized + Copy {
    /// Apply Q from QR factorization to a matrix C.
    unsafe fn unmqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Trait implementations for f32
// ============================================================================

impl OrgqrType for f32 {
    unsafe fn orgqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_sorgqr(cast_handle(handle), m, n, k, A, lda, ipiv)
    }
}

impl OrmqrType for f32 {
    unsafe fn ormqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sormqr(
            cast_handle(handle),
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    }
}

// ============================================================================
// Trait implementations for f64
// ============================================================================

impl OrgqrType for f64 {
    unsafe fn orgqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_dorgqr(cast_handle(handle), m, n, k, A, lda, ipiv)
    }
}

impl OrmqrType for f64 {
    unsafe fn ormqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dormqr(
            cast_handle(handle),
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    }
}

// ============================================================================
// Trait implementations for Complex32
// ============================================================================

impl UngqrType for Complex32 {
    unsafe fn ungqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_cungqr(
            cast_handle(handle),
            m,
            n,
            k,
            A,
            lda,
            ipiv,
        )
    }
}

impl UnmqrType for Complex32 {
    unsafe fn unmqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cunmqr(
            cast_handle(handle),
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    }
}

// ============================================================================
// Trait implementations for Complex64
// ============================================================================

impl UngqrType for Complex64 {
    unsafe fn ungqr(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_zungqr(
            cast_handle(handle),
            m,
            n,
            k,
            A,
            lda,
            ipiv,
        )
    }
}

impl UnmqrType for Complex64 {
    unsafe fn unmqr(
        handle: RocblasHandle,
        side: rocblas_ffi::rocblas_side,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        k: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
        C: *mut Self,
        ldc: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zunmqr(
            cast_handle(handle),
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    }
}

// ============================================================================
// Public API functions for real matrices
// ============================================================================

/// Generates the orthogonal matrix Q from QR factorization.
///
/// Generates an m-by-n real matrix Q with orthonormal columns,
/// defined as the first n columns of the product of k elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k)
/// as returned by geqrf.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of Q (m >= 0)
/// * `n` - Number of columns of Q (m >= n >= 0)
/// * `k` - Number of elementary reflectors (n >= k >= 0)
/// * `A` - Device pointer to matrix containing reflectors (modified to contain Q)
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to Householder scalars from geqrf (k elements)
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
///
/// # Example
/// ```rust,no_run
/// use rocm_rs::{rocblas::Handle, rocsolver};
///
/// let handle = Handle::new().unwrap();
/// let m = 4i32;
/// let n = 3i32;
/// let k = 3i32;
/// // A and ipiv would be device pointers from geqrf
/// // rocsolver::orgqr(&handle, m, n, k, A, m, ipiv).unwrap();
/// ```
#[inline]
pub fn orgqr<T: OrgqrType>(
    handle: &Handle,
    m: i32,
    n: i32,
    k: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut T,
) -> Result<()> {
    let status = unsafe { T::orgqr(handle.as_raw(), m, n, k, A, lda, ipiv) };
    Error::from_status(status)
}

/// Applies Q from QR factorization to a matrix C.
///
/// Multiplies matrix C by the orthogonal matrix Q from geqrf:
/// - C = Q * C   (side=Left, trans=None)
/// - C = Q^T * C (side=Left, trans=Transpose)
/// - C = C * Q   (side=Right, trans=None)
/// - C = C * Q^T (side=Right, trans=Transpose)
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `side` - Apply Q from left or right
/// * `trans` - Whether to transpose Q
/// * `m` - Number of rows of C
/// * `n` - Number of columns of C
/// * `k` - Number of elementary reflectors
/// * `A` - Device pointer to matrix containing reflectors
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to Householder scalars
/// * `C` - Device pointer to matrix C (modified)
/// * `ldc` - Leading dimension of C
///
/// # Example
/// ```rust,no_run
/// use rocm_rs::{rocblas::{Handle, types::{Side, Operation}}, rocsolver};
///
/// let handle = Handle::new().unwrap();
/// // Apply Q from left with no transpose: C = Q * C
/// // rocsolver::ormqr(&handle, Side::Left, Operation::None, m, n, k, A, lda, ipiv, C, ldc).unwrap();
/// ```
#[inline]
pub fn ormqr<T: OrmqrType>(
    handle: &Handle,
    side: Side,
    trans: Operation,
    m: i32,
    n: i32,
    k: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut T,
    C: *mut T,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::ormqr(
            handle.as_raw(),
            side.into(),
            trans.into(),
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    };
    Error::from_status(status)
}

// ============================================================================
// Public API functions for complex matrices
// ============================================================================

/// Generates the unitary matrix Q from QR factorization (complex version).
///
/// Generates an m-by-n complex matrix Q with orthonormal columns,
/// defined as the first n columns of the product of k elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k)
/// as returned by geqrf.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of Q (m >= 0)
/// * `n` - Number of columns of Q (m >= n >= 0)
/// * `k` - Number of elementary reflectors (n >= k >= 0)
/// * `A` - Device pointer to matrix containing reflectors (modified to contain Q)
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to Householder scalars from geqrf (k elements)
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
#[inline]
pub fn ungqr<T: UngqrType>(
    handle: &Handle,
    m: i32,
    n: i32,
    k: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut T,
) -> Result<()> {
    let status = unsafe { T::ungqr(handle.as_raw(), m, n, k, A, lda, ipiv) };
    Error::from_status(status)
}

/// Applies Q from QR factorization to a matrix C (complex version).
///
/// Multiplies matrix C by the unitary matrix Q from geqrf:
/// - C = Q * C   (side=Left, trans=None)
/// - C = Q^H * C (side=Left, trans=ConjugateTranspose)
/// - C = C * Q   (side=Right, trans=None)
/// - C = C * Q^H (side=Right, trans=ConjugateTranspose)
///
/// Note: For complex matrices, use ConjugateTranspose instead of Transpose.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `side` - Apply Q from left or right
/// * `trans` - Whether to apply conjugate transpose of Q
/// * `m` - Number of rows of C
/// * `n` - Number of columns of C
/// * `k` - Number of elementary reflectors
/// * `A` - Device pointer to matrix containing reflectors
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to Householder scalars
/// * `C` - Device pointer to matrix C (modified)
/// * `ldc` - Leading dimension of C
#[inline]
pub fn unmqr<T: UnmqrType>(
    handle: &Handle,
    side: Side,
    trans: Operation,
    m: i32,
    n: i32,
    k: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut T,
    C: *mut T,
    ldc: i32,
) -> Result<()> {
    let status = unsafe {
        T::unmqr(
            handle.as_raw(),
            side.into(),
            trans.into(),
            m,
            n,
            k,
            A,
            lda,
            ipiv,
            C,
            ldc,
        )
    };
    Error::from_status(status)
}
