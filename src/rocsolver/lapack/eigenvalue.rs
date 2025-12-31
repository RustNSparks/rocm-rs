// src/rocsolver/lapack/eigenvalue.rs
//! Eigenvalue computations.
//!
//! This module provides safe wrappers for eigenvalue decomposition:
//!
//! - [`syev`] - Eigenvalues/vectors of a real symmetric matrix
//! - [`heev`] - Eigenvalues/vectors of a complex Hermitian matrix

use crate::rocblas::Handle;
use crate::rocblas::ffi as rocblas_ffi;
use crate::rocsolver::bindings;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::types::{Complex32, Complex64, Evect, Fill};

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

/// Trait for types that support symmetric eigenvalue decomposition (syev).
pub trait SyevType: Sized + Copy {
    /// Compute eigenvalues and optionally eigenvectors of a symmetric matrix.
    unsafe fn syev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self,
        E: *mut Self,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Batched syev.
    unsafe fn syev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched syev.
    unsafe fn syev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support Hermitian eigenvalue decomposition (heev).
pub trait HeevType: Sized + Copy {
    /// The real type for eigenvalues.
    type RealType: Copy;

    /// Compute eigenvalues and optionally eigenvectors of a Hermitian matrix.
    unsafe fn heev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Batched heev.
    unsafe fn heev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched heev.
    unsafe fn heev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Trait implementations for f32
// ============================================================================

impl SyevType for f32 {
    unsafe fn syev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self,
        E: *mut Self,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_ssyev(cast_handle(handle), evect, uplo, n, A, lda, D, E, info)
    }

    unsafe fn syev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_ssyev_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }

    unsafe fn syev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_ssyev_strided_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for f64
// ============================================================================

impl SyevType for f64 {
    unsafe fn syev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self,
        E: *mut Self,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dsyev(cast_handle(handle), evect, uplo, n, A, lda, D, E, info)
    }

    unsafe fn syev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dsyev_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }

    unsafe fn syev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self,
        stride_d: i64,
        E: *mut Self,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dsyev_strided_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex32
// ============================================================================

impl HeevType for Complex32 {
    type RealType = f32;

    unsafe fn heev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cheev(cast_handle(handle), evect, uplo, n, A, lda, D, E, info)
    }

    unsafe fn heev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cheev_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }

    unsafe fn heev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cheev_strided_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex64
// ============================================================================

impl HeevType for Complex64 {
    type RealType = f64;

    unsafe fn heev(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zheev(cast_handle(handle), evect, uplo, n, A, lda, D, E, info)
    }

    unsafe fn heev_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zheev_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }

    unsafe fn heev_strided_batched(
        handle: RocblasHandle,
        evect: bindings::rocblas_evect,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zheev_strided_batched(
            cast_handle(handle),
            evect,
            uplo,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Computes eigenvalues and optionally eigenvectors of a real symmetric matrix.
///
/// The eigenvalue decomposition is:
///   A = V * D * V^T
/// where D is diagonal containing eigenvalues, and V is orthogonal containing eigenvectors.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `evect` - Specifies whether to compute eigenvectors (Original, Tridiagonal, None)
/// * `uplo` - Specifies whether upper or lower triangle of A is stored
/// * `n` - Order of matrix A
/// * `A` - Device pointer to n-by-n symmetric matrix (modified; contains eigenvectors if evect=Original)
/// * `lda` - Leading dimension of A
/// * `D` - Device pointer to eigenvalues (n elements, ascending order)
/// * `E` - Device pointer to workspace (n-1 elements)
/// * `info` - Device pointer to info value
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
#[inline]
pub fn syev<T: SyevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    D: *mut T,
    E: *mut T,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe {
        T::syev(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            D,
            E,
            info,
        )
    };
    Error::from_status(status)
}

/// Batched version of syev.
#[inline]
pub fn syev_batched<T: SyevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *const *mut T,
    lda: i32,
    D: *mut T,
    stride_d: i64,
    E: *mut T,
    stride_e: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::syev_batched(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of syev.
#[inline]
pub fn syev_strided_batched<T: SyevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    D: *mut T,
    stride_d: i64,
    E: *mut T,
    stride_e: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::syev_strided_batched(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Computes eigenvalues and optionally eigenvectors of a complex Hermitian matrix.
///
/// The eigenvalue decomposition is:
///   A = V * D * V^H
/// where D is real diagonal containing eigenvalues, and V is unitary containing eigenvectors.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `evect` - Specifies whether to compute eigenvectors
/// * `uplo` - Specifies whether upper or lower triangle of A is stored
/// * `n` - Order of matrix A
/// * `A` - Device pointer to n-by-n Hermitian matrix
/// * `lda` - Leading dimension of A
/// * `D` - Device pointer to real eigenvalues (n elements)
/// * `E` - Device pointer to real workspace (n-1 elements)
/// * `info` - Device pointer to info value
#[inline]
pub fn heev<T: HeevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    D: *mut T::RealType,
    E: *mut T::RealType,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe {
        T::heev(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            D,
            E,
            info,
        )
    };
    Error::from_status(status)
}

/// Batched version of heev.
#[inline]
pub fn heev_batched<T: HeevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *const *mut T,
    lda: i32,
    D: *mut T::RealType,
    stride_d: i64,
    E: *mut T::RealType,
    stride_e: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::heev_batched(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of heev.
#[inline]
pub fn heev_strided_batched<T: HeevType>(
    handle: &Handle,
    evect: Evect,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    D: *mut T::RealType,
    stride_d: i64,
    E: *mut T::RealType,
    stride_e: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::heev_strided_batched(
            handle.as_raw(),
            evect.into(),
            uplo.into(),
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}
