// src/rocsolver/lapack/decompositions.rs
//! Matrix decomposition/factorization operations.
//!
//! This module provides safe wrappers for matrix factorization routines:
//!
//! - **QR factorization**: [`geqrf`], [`geqrf_batched`], [`geqrf_strided_batched`]
//! - **LU factorization**: [`getrf`], [`getrf_batched`], [`getrf_strided_batched`]
//! - **LU without pivoting**: [`getrf_npvt`], [`getrf_npvt_batched`], [`getrf_npvt_strided_batched`]
//! - **Cholesky factorization**: [`potrf`], [`potrf_batched`], [`potrf_strided_batched`]
//! - **Bidiagonal reduction**: [`gebrd`], [`gebrd_batched`], [`gebrd_strided_batched`]

use crate::rocblas::Handle;
use crate::rocblas::ffi as rocblas_ffi;
use crate::rocsolver::bindings;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::types::{Complex32, Complex64, Fill};

// Type alias for handle - we use rocblas handle but need to cast for rocsolver bindings
type RocblasHandle = rocblas_ffi::rocblas_handle;
type RocblasStatus = rocblas_ffi::rocblas_status;

/// Cast rocblas handle to rocsolver bindings handle type.
/// Both are pointers to the same opaque C struct, so this is safe.
#[inline]
fn cast_handle(handle: RocblasHandle) -> bindings::rocblas_handle {
    handle as bindings::rocblas_handle
}

// Complex type casts - both rocblas and rocsolver bindings define these identically
type BindingsComplex32 = bindings::rocblas_float_complex;
type BindingsComplex64 = bindings::rocblas_double_complex;

// ============================================================================
// Type traits for generic implementations
// ============================================================================

/// Trait for types that support QR factorization (geqrf).
pub trait GeqrfType: Sized + Copy {
    /// Perform QR factorization.
    unsafe fn geqrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus;

    /// Perform batched QR factorization.
    unsafe fn geqrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform strided batched QR factorization.
    unsafe fn geqrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support LU factorization (getrf).
pub trait GetrfType: Sized + Copy {
    /// Perform LU factorization with partial pivoting.
    unsafe fn getrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Perform batched LU factorization.
    unsafe fn getrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform strided batched LU factorization.
    unsafe fn getrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform LU factorization without pivoting.
    unsafe fn getrf_npvt(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Perform batched LU factorization without pivoting.
    unsafe fn getrf_npvt_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform strided batched LU factorization without pivoting.
    unsafe fn getrf_npvt_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support Cholesky factorization (potrf).
pub trait PotrfType: Sized + Copy {
    /// Perform Cholesky factorization.
    unsafe fn potrf(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Perform batched Cholesky factorization.
    unsafe fn potrf_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform strided batched Cholesky factorization.
    unsafe fn potrf_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support bidiagonal reduction (gebrd).
pub trait GebrdType: Sized + Copy {
    /// The real type for diagonal/off-diagonal elements.
    type RealType: Copy;

    /// Perform bidiagonal reduction.
    unsafe fn gebrd(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        tauq: *mut Self,
        taup: *mut Self,
    ) -> RocblasStatus;

    /// Perform batched bidiagonal reduction.
    unsafe fn gebrd_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Perform strided batched bidiagonal reduction.
    unsafe fn gebrd_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Trait implementations for f32
// ============================================================================

impl GeqrfType for f32 {
    unsafe fn geqrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_sgeqrf(cast_handle(handle), m, n, A, lda, ipiv)
    }

    unsafe fn geqrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgeqrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            batch_count,
        )
    }

    unsafe fn geqrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgeqrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            batch_count,
        )
    }
}

impl GetrfType for f32 {
    unsafe fn getrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf(cast_handle(handle), m, n, A, lda, ipiv, info)
    }

    unsafe fn getrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf_npvt(cast_handle(handle), m, n, A, lda, info)
    }

    unsafe fn getrf_npvt_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf_npvt_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrf_npvt_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl PotrfType for f32 {
    unsafe fn potrf(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_spotrf(cast_handle(handle), uplo, n, A, lda, info)
    }

    unsafe fn potrf_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_spotrf_batched(cast_handle(handle), uplo, n, A, lda, info, batch_count)
    }

    unsafe fn potrf_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_spotrf_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl GebrdType for f32 {
    type RealType = f32;

    unsafe fn gebrd(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        tauq: *mut Self,
        taup: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_sgebrd(cast_handle(handle), m, n, A, lda, D, E, tauq, taup)
    }

    unsafe fn gebrd_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgebrd_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }

    unsafe fn gebrd_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgebrd_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for f64
// ============================================================================

impl GeqrfType for f64 {
    unsafe fn geqrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_dgeqrf(cast_handle(handle), m, n, A, lda, ipiv)
    }

    unsafe fn geqrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgeqrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            batch_count,
        )
    }

    unsafe fn geqrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgeqrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            batch_count,
        )
    }
}

impl GetrfType for f64 {
    unsafe fn getrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf(cast_handle(handle), m, n, A, lda, ipiv, info)
    }

    unsafe fn getrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf_npvt(cast_handle(handle), m, n, A, lda, info)
    }

    unsafe fn getrf_npvt_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf_npvt_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrf_npvt_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl PotrfType for f64 {
    unsafe fn potrf(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dpotrf(cast_handle(handle), uplo, n, A, lda, info)
    }

    unsafe fn potrf_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dpotrf_batched(cast_handle(handle), uplo, n, A, lda, info, batch_count)
    }

    unsafe fn potrf_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dpotrf_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl GebrdType for f64 {
    type RealType = f64;

    unsafe fn gebrd(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        tauq: *mut Self,
        taup: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_dgebrd(cast_handle(handle), m, n, A, lda, D, E, tauq, taup)
    }

    unsafe fn gebrd_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgebrd_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }

    unsafe fn gebrd_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgebrd_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex32
// ============================================================================

impl GeqrfType for Complex32 {
    unsafe fn geqrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_cgeqrf(cast_handle(handle), m, n, A, lda, ipiv)
    }

    unsafe fn geqrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgeqrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            batch_count,
        )
    }

    unsafe fn geqrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgeqrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            batch_count,
        )
    }
}

impl GetrfType for Complex32 {
    unsafe fn getrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf(cast_handle(handle), m, n, A, lda, ipiv, info)
    }

    unsafe fn getrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf_npvt(cast_handle(handle), m, n, A, lda, info)
    }

    unsafe fn getrf_npvt_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf_npvt_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrf_npvt_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl PotrfType for Complex32 {
    unsafe fn potrf(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cpotrf(cast_handle(handle), uplo, n, A, lda, info)
    }

    unsafe fn potrf_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cpotrf_batched(cast_handle(handle), uplo, n, A, lda, info, batch_count)
    }

    unsafe fn potrf_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cpotrf_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl GebrdType for Complex32 {
    type RealType = f32;

    unsafe fn gebrd(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        tauq: *mut Self,
        taup: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_cgebrd(cast_handle(handle), m, n, A, lda, D, E, tauq, taup)
    }

    unsafe fn gebrd_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgebrd_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }

    unsafe fn gebrd_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgebrd_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex64
// ============================================================================

impl GeqrfType for Complex64 {
    unsafe fn geqrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_zgeqrf(cast_handle(handle), m, n, A, lda, ipiv)
    }

    unsafe fn geqrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgeqrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            batch_count,
        )
    }

    unsafe fn geqrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut Self,
        stride_p: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgeqrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            batch_count,
        )
    }
}

impl GetrfType for Complex64 {
    unsafe fn getrf(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf(cast_handle(handle), m, n, A, lda, ipiv, info)
    }

    unsafe fn getrf_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf_npvt(cast_handle(handle), m, n, A, lda, info)
    }

    unsafe fn getrf_npvt_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf_npvt_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            info,
            batch_count,
        )
    }

    unsafe fn getrf_npvt_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrf_npvt_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl PotrfType for Complex64 {
    unsafe fn potrf(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zpotrf(cast_handle(handle), uplo, n, A, lda, info)
    }

    unsafe fn potrf_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zpotrf_batched(cast_handle(handle), uplo, n, A, lda, info, batch_count)
    }

    unsafe fn potrf_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zpotrf_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    }
}

impl GebrdType for Complex64 {
    type RealType = f64;

    unsafe fn gebrd(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        E: *mut Self::RealType,
        tauq: *mut Self,
        taup: *mut Self,
    ) -> RocblasStatus {
        bindings::rocsolver_zgebrd(cast_handle(handle), m, n, A, lda, D, E, tauq, taup)
    }

    unsafe fn gebrd_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *const *mut Self,
        lda: i32,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgebrd_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }

    unsafe fn gebrd_strided_batched(
        handle: RocblasHandle,
        m: i32,
        n: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        D: *mut Self::RealType,
        stride_d: i64,
        E: *mut Self::RealType,
        stride_e: i64,
        tauq: *mut Self,
        stride_tauq: i64,
        taup: *mut Self,
        stride_taup: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgebrd_strided_batched(
            cast_handle(handle),
            m,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Computes the QR factorization of a general m-by-n matrix A.
///
/// The factorization has the form:
///   A = Q * R
/// where Q is an orthogonal/unitary matrix and R is upper triangular.
///
/// The matrix Q is represented as a product of elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k), where k = min(m,n)
///
/// Each H(i) has the form:
///   H(i) = I - tau[i] * v * v^H
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of matrix A (m >= 0)
/// * `n` - Number of columns of matrix A (n >= 0)
/// * `A` - Device pointer to m-by-n matrix (modified in-place to contain R)
/// * `lda` - Leading dimension of A (lda >= max(1,m))
/// * `ipiv` - Device pointer to array of Householder scalars (min(m,n) elements)
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
///
/// # Example
/// ```rust,no_run
/// use rocm_rs::{hip::DeviceMemory, rocblas::Handle, rocsolver};
///
/// let handle = Handle::new().unwrap();
/// let m = 4;
/// let n = 3;
/// let mut A = DeviceMemory::<f64>::new((m * n) as usize).unwrap();
/// let mut tau = DeviceMemory::<f64>::new(n as usize).unwrap();
///
/// // Initialize A with matrix data...
///
/// rocsolver::geqrf(&handle, m, n, A.as_mut_ptr() as *mut f64, m, tau.as_mut_ptr() as *mut f64).unwrap();
/// ```
#[inline]
pub fn geqrf<T: GeqrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut T,
) -> Result<()> {
    let status = unsafe { T::geqrf(handle.as_raw(), m, n, A, lda, ipiv) };
    Error::from_status(status)
}

/// Computes the batched QR factorization of multiple m-by-n matrices.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of each matrix A
/// * `n` - Number of columns of each matrix A
/// * `A` - Array of device pointers to m-by-n matrices
/// * `lda` - Leading dimension of each A
/// * `ipiv` - Device pointer to Householder scalars (min(m,n) * batch_count elements)
/// * `stride_p` - Stride between ipiv arrays
/// * `batch_count` - Number of matrices
#[inline]
pub fn geqrf_batched<T: GeqrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *const *mut T,
    lda: i32,
    ipiv: *mut T,
    stride_p: i64,
    batch_count: i32,
) -> Result<()> {
    let status =
        unsafe { T::geqrf_batched(handle.as_raw(), m, n, A, lda, ipiv, stride_p, batch_count) };
    Error::from_status(status)
}

/// Computes the strided batched QR factorization of multiple m-by-n matrices.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of each matrix A
/// * `n` - Number of columns of each matrix A
/// * `A` - Device pointer to first matrix (strided)
/// * `lda` - Leading dimension of each A
/// * `stride_a` - Stride between A matrices
/// * `ipiv` - Device pointer to Householder scalars
/// * `stride_p` - Stride between ipiv arrays
/// * `batch_count` - Number of matrices
#[inline]
pub fn geqrf_strided_batched<T: GeqrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    ipiv: *mut T,
    stride_p: i64,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::geqrf_strided_batched(
            handle.as_raw(),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Computes the LU factorization of a general m-by-n matrix A with partial pivoting.
///
/// The factorization has the form:
///   A = P * L * U
/// where P is a permutation matrix, L is lower triangular with unit diagonal,
/// and U is upper triangular.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of matrix A (m >= 0)
/// * `n` - Number of columns of matrix A (n >= 0)
/// * `A` - Device pointer to m-by-n matrix (modified in-place)
/// * `lda` - Leading dimension of A (lda >= max(1,m))
/// * `ipiv` - Device pointer to pivot indices (min(m,n) elements)
/// * `info` - Device pointer to info value (0 = success, i > 0 = U(i,i) is zero)
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
#[inline]
pub fn getrf<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe { T::getrf(handle.as_raw(), m, n, A, lda, ipiv, info) };
    Error::from_status(status)
}

/// Computes the batched LU factorization of multiple matrices with partial pivoting.
#[inline]
pub fn getrf_batched<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *const *mut T,
    lda: i32,
    ipiv: *mut i32,
    stride_p: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::getrf_batched(
            handle.as_raw(),
            m,
            n,
            A,
            lda,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Computes the strided batched LU factorization of multiple matrices.
#[inline]
pub fn getrf_strided_batched<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    ipiv: *mut i32,
    stride_p: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::getrf_strided_batched(
            handle.as_raw(),
            m,
            n,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Computes the LU factorization without pivoting.
///
/// This variant does not use row interchanges, which can be faster but
/// may fail for singular or nearly-singular matrices.
#[inline]
pub fn getrf_npvt<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe { T::getrf_npvt(handle.as_raw(), m, n, A, lda, info) };
    Error::from_status(status)
}

/// Computes the batched LU factorization without pivoting.
#[inline]
pub fn getrf_npvt_batched<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *const *mut T,
    lda: i32,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe { T::getrf_npvt_batched(handle.as_raw(), m, n, A, lda, info, batch_count) };
    Error::from_status(status)
}

/// Computes the strided batched LU factorization without pivoting.
#[inline]
pub fn getrf_npvt_strided_batched<T: GetrfType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::getrf_npvt_strided_batched(handle.as_raw(), m, n, A, lda, stride_a, info, batch_count)
    };
    Error::from_status(status)
}

/// Computes the Cholesky factorization of a symmetric/Hermitian positive-definite matrix.
///
/// The factorization has the form:
///   A = U^H * U  if uplo = Upper
///   A = L * L^H  if uplo = Lower
/// where U is upper triangular and L is lower triangular.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `uplo` - Specifies whether upper or lower triangle of A is stored
/// * `n` - Order of matrix A (n >= 0)
/// * `A` - Device pointer to n-by-n matrix (modified in-place)
/// * `lda` - Leading dimension of A (lda >= max(1,n))
/// * `info` - Device pointer to info value (0 = success, i > 0 = leading minor i not positive definite)
#[inline]
pub fn potrf<T: PotrfType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe { T::potrf(handle.as_raw(), uplo.into(), n, A, lda, info) };
    Error::from_status(status)
}

/// Computes the batched Cholesky factorization.
#[inline]
pub fn potrf_batched<T: PotrfType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: *const *mut T,
    lda: i32,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status =
        unsafe { T::potrf_batched(handle.as_raw(), uplo.into(), n, A, lda, info, batch_count) };
    Error::from_status(status)
}

/// Computes the strided batched Cholesky factorization.
#[inline]
pub fn potrf_strided_batched<T: PotrfType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::potrf_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            A,
            lda,
            stride_a,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Reduces a general matrix to bidiagonal form.
///
/// The reduction is:
///   A = Q * B * P^H
/// where Q and P are orthogonal/unitary, and B is bidiagonal.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `A` - Device pointer to m-by-n matrix (modified in-place)
/// * `lda` - Leading dimension of A
/// * `D` - Device pointer to diagonal of B (min(m,n) elements)
/// * `E` - Device pointer to off-diagonal of B (min(m,n)-1 elements)
/// * `tauq` - Device pointer to Q scalars (min(m,n) elements)
/// * `taup` - Device pointer to P scalars (min(m,n) elements)
#[inline]
pub fn gebrd<T: GebrdType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    D: *mut T::RealType,
    E: *mut T::RealType,
    tauq: *mut T,
    taup: *mut T,
) -> Result<()> {
    let status = unsafe { T::gebrd(handle.as_raw(), m, n, A, lda, D, E, tauq, taup) };
    Error::from_status(status)
}

/// Computes the batched bidiagonal reduction.
#[inline]
pub fn gebrd_batched<T: GebrdType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *const *mut T,
    lda: i32,
    D: *mut T::RealType,
    stride_d: i64,
    E: *mut T::RealType,
    stride_e: i64,
    tauq: *mut T,
    stride_tauq: i64,
    taup: *mut T,
    stride_taup: i64,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gebrd_batched(
            handle.as_raw(),
            m,
            n,
            A,
            lda,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Computes the strided batched bidiagonal reduction.
#[inline]
pub fn gebrd_strided_batched<T: GebrdType>(
    handle: &Handle,
    m: i32,
    n: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    D: *mut T::RealType,
    stride_d: i64,
    E: *mut T::RealType,
    stride_e: i64,
    tauq: *mut T,
    stride_tauq: i64,
    taup: *mut T,
    stride_taup: i64,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gebrd_strided_batched(
            handle.as_raw(),
            m,
            n,
            A,
            lda,
            stride_a,
            D,
            stride_d,
            E,
            stride_e,
            tauq,
            stride_tauq,
            taup,
            stride_taup,
            batch_count,
        )
    };
    Error::from_status(status)
}
