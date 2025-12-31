// src/rocsolver/lapack/solvers.rs
//! Linear system solvers.
//!
//! This module provides safe wrappers for solving linear systems of equations:
//!
//! - **General solver**: [`gesv`] - Solves A*X = B using LU factorization
//! - **Triangular solver**: [`getrs`] - Solves using pre-computed LU factors
//! - **Positive definite solver**: [`posv`] - Solves A*X = B using Cholesky
//! - **Least squares solver**: [`gels`] - Solves overdetermined/underdetermined systems

use crate::rocblas::Handle;
use crate::rocblas::ffi as rocblas_ffi;
use crate::rocsolver::bindings;
use crate::rocsolver::error::{Error, Result};
use crate::rocsolver::types::{Complex32, Complex64, Fill, Operation};

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

/// Trait for types that support gesv (general linear solver).
pub trait GesvType: Sized + Copy {
    /// Solve A*X = B using LU factorization.
    unsafe fn gesv(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Batched gesv.
    unsafe fn gesv_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched gesv.
    unsafe fn gesv_strided_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support getrs (solve with LU factors).
pub trait GetrsType: Sized + Copy {
    /// Solve A*X = B using pre-computed LU factorization.
    unsafe fn getrs(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *const i32,
        B: *mut Self,
        ldb: i32,
    ) -> RocblasStatus;

    /// Batched getrs.
    unsafe fn getrs_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *const i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched getrs.
    unsafe fn getrs_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *const i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support posv (positive definite solver).
pub trait PosvType: Sized + Copy {
    /// Solve A*X = B for positive definite A using Cholesky.
    unsafe fn posv(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Batched posv.
    unsafe fn posv_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched posv.
    unsafe fn posv_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

/// Trait for types that support gels (least squares solver).
pub trait GelsType: Sized + Copy {
    /// Solve overdetermined or underdetermined linear systems.
    unsafe fn gels(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus;

    /// Batched gels.
    unsafe fn gels_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;

    /// Strided batched gels.
    unsafe fn gels_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus;
}

// ============================================================================
// Trait implementations for f32
// ============================================================================

impl GesvType for f32 {
    unsafe fn gesv(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgesv(cast_handle(handle), n, nrhs, A, lda, ipiv, B, ldb, info)
    }

    unsafe fn gesv_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgesv_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gesv_strided_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgesv_strided_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GetrsType for f32 {
    unsafe fn getrs(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *const i32,
        B: *mut Self,
        ldb: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrs(cast_handle(handle), trans, n, nrhs, A, lda, ipiv, B, ldb)
    }

    unsafe fn getrs_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *const i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrs_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            batch_count,
        )
    }

    unsafe fn getrs_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *const i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgetrs_strided_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            batch_count,
        )
    }
}

impl PosvType for f32 {
    unsafe fn posv(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sposv(cast_handle(handle), uplo, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn posv_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sposv_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn posv_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sposv_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GelsType for f32 {
    unsafe fn gels(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgels(cast_handle(handle), trans, m, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn gels_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgels_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gels_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_sgels_strided_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for f64
// ============================================================================

impl GesvType for f64 {
    unsafe fn gesv(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgesv(cast_handle(handle), n, nrhs, A, lda, ipiv, B, ldb, info)
    }

    unsafe fn gesv_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgesv_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gesv_strided_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgesv_strided_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GetrsType for f64 {
    unsafe fn getrs(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *const i32,
        B: *mut Self,
        ldb: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrs(cast_handle(handle), trans, n, nrhs, A, lda, ipiv, B, ldb)
    }

    unsafe fn getrs_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *const i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrs_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            batch_count,
        )
    }

    unsafe fn getrs_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *const i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgetrs_strided_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            batch_count,
        )
    }
}

impl PosvType for f64 {
    unsafe fn posv(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dposv(cast_handle(handle), uplo, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn posv_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dposv_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn posv_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dposv_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GelsType for f64 {
    unsafe fn gels(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgels(cast_handle(handle), trans, m, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn gels_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgels_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gels_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_dgels_strided_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex32
// ============================================================================

impl GesvType for Complex32 {
    unsafe fn gesv(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgesv(cast_handle(handle), n, nrhs, A, lda, ipiv, B, ldb, info)
    }

    unsafe fn gesv_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgesv_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gesv_strided_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgesv_strided_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GetrsType for Complex32 {
    unsafe fn getrs(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *const i32,
        B: *mut Self,
        ldb: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrs(cast_handle(handle), trans, n, nrhs, A, lda, ipiv, B, ldb)
    }

    unsafe fn getrs_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *const i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrs_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            batch_count,
        )
    }

    unsafe fn getrs_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *const i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgetrs_strided_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            batch_count,
        )
    }
}

impl PosvType for Complex32 {
    unsafe fn posv(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cposv(cast_handle(handle), uplo, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn posv_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cposv_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn posv_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cposv_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GelsType for Complex32 {
    unsafe fn gels(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgels(cast_handle(handle), trans, m, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn gels_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgels_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gels_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_cgels_strided_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Trait implementations for Complex64
// ============================================================================

impl GesvType for Complex64 {
    unsafe fn gesv(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *mut i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgesv(cast_handle(handle), n, nrhs, A, lda, ipiv, B, ldb, info)
    }

    unsafe fn gesv_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *mut i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgesv_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gesv_strided_batched(
        handle: RocblasHandle,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *mut i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgesv_strided_batched(
            cast_handle(handle),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GetrsType for Complex64 {
    unsafe fn getrs(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        ipiv: *const i32,
        B: *mut Self,
        ldb: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrs(cast_handle(handle), trans, n, nrhs, A, lda, ipiv, B, ldb)
    }

    unsafe fn getrs_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        ipiv: *const i32,
        stride_p: i64,
        B: *const *mut Self,
        ldb: i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrs_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            batch_count,
        )
    }

    unsafe fn getrs_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        ipiv: *const i32,
        stride_p: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgetrs_strided_batched(
            cast_handle(handle),
            trans,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            batch_count,
        )
    }
}

impl PosvType for Complex64 {
    unsafe fn posv(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zposv(cast_handle(handle), uplo, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn posv_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zposv_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn posv_strided_batched(
        handle: RocblasHandle,
        uplo: rocblas_ffi::rocblas_fill,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zposv_strided_batched(
            cast_handle(handle),
            uplo,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

impl GelsType for Complex64 {
    unsafe fn gels(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        B: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgels(cast_handle(handle), trans, m, n, nrhs, A, lda, B, ldb, info)
    }

    unsafe fn gels_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *const *mut Self,
        lda: i32,
        B: *const *mut Self,
        ldb: i32,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgels_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    }

    unsafe fn gels_strided_batched(
        handle: RocblasHandle,
        trans: rocblas_ffi::rocblas_operation,
        m: i32,
        n: i32,
        nrhs: i32,
        A: *mut Self,
        lda: i32,
        stride_a: i64,
        B: *mut Self,
        ldb: i32,
        stride_b: i64,
        info: *mut i32,
        batch_count: i32,
    ) -> RocblasStatus {
        bindings::rocsolver_zgels_strided_batched(
            cast_handle(handle),
            trans,
            m,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Solves a general system of linear equations A*X = B.
///
/// Uses LU factorization with partial pivoting to solve the system.
/// On exit, A contains the LU factors and B contains the solution X.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `n` - Order of matrix A (n >= 0)
/// * `nrhs` - Number of right-hand sides (columns of B)
/// * `A` - Device pointer to n-by-n matrix (modified to contain LU factors)
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to pivot indices (n elements)
/// * `B` - Device pointer to n-by-nrhs matrix (modified to contain solution)
/// * `ldb` - Leading dimension of B
/// * `info` - Device pointer to info value
///
/// # Returns
/// `Ok(())` on success, or an error if the operation failed.
#[inline]
pub fn gesv<T: GesvType>(
    handle: &Handle,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut i32,
    B: *mut T,
    ldb: i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe { T::gesv(handle.as_raw(), n, nrhs, A, lda, ipiv, B, ldb, info) };
    Error::from_status(status)
}

/// Batched version of gesv.
#[inline]
pub fn gesv_batched<T: GesvType>(
    handle: &Handle,
    n: i32,
    nrhs: i32,
    A: *const *mut T,
    lda: i32,
    ipiv: *mut i32,
    stride_p: i64,
    B: *const *mut T,
    ldb: i32,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gesv_batched(
            handle.as_raw(),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of gesv.
#[inline]
pub fn gesv_strided_batched<T: GesvType>(
    handle: &Handle,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    ipiv: *mut i32,
    stride_p: i64,
    B: *mut T,
    ldb: i32,
    stride_b: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gesv_strided_batched(
            handle.as_raw(),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Solves a system of linear equations using pre-computed LU factorization.
///
/// Solves one of the following systems:
/// - A*X = B   (trans = None)
/// - A^T*X = B (trans = Transpose)
/// - A^H*X = B (trans = ConjugateTranspose)
///
/// where A has been factorized by getrf.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `trans` - Specifies the form of the system
/// * `n` - Order of matrix A
/// * `nrhs` - Number of right-hand sides
/// * `A` - Device pointer to LU factors (from getrf)
/// * `lda` - Leading dimension of A
/// * `ipiv` - Device pointer to pivot indices (from getrf)
/// * `B` - Device pointer to right-hand side (modified to contain solution)
/// * `ldb` - Leading dimension of B
#[inline]
pub fn getrs<T: GetrsType>(
    handle: &Handle,
    trans: Operation,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    ipiv: *mut i32,
    B: *mut T,
    ldb: i32,
) -> Result<()> {
    let status = unsafe { T::getrs(handle.as_raw(), trans.into(), n, nrhs, A, lda, ipiv, B, ldb) };
    Error::from_status(status)
}

/// Batched version of getrs.
#[inline]
pub fn getrs_batched<T: GetrsType>(
    handle: &Handle,
    trans: Operation,
    n: i32,
    nrhs: i32,
    A: *const *mut T,
    lda: i32,
    ipiv: *mut i32,
    stride_p: i64,
    B: *const *mut T,
    ldb: i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::getrs_batched(
            handle.as_raw(),
            trans.into(),
            n,
            nrhs,
            A,
            lda,
            ipiv,
            stride_p,
            B,
            ldb,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of getrs.
#[inline]
pub fn getrs_strided_batched<T: GetrsType>(
    handle: &Handle,
    trans: Operation,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    ipiv: *mut i32,
    stride_p: i64,
    B: *mut T,
    ldb: i32,
    stride_b: i64,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::getrs_strided_batched(
            handle.as_raw(),
            trans.into(),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            ipiv,
            stride_p,
            B,
            ldb,
            stride_b,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Solves a symmetric/Hermitian positive-definite system A*X = B.
///
/// Uses Cholesky factorization to solve the system.
/// On exit, A contains the Cholesky factor and B contains the solution.
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `uplo` - Specifies upper or lower triangular storage
/// * `n` - Order of matrix A
/// * `nrhs` - Number of right-hand sides
/// * `A` - Device pointer to n-by-n SPD matrix (modified)
/// * `lda` - Leading dimension of A
/// * `B` - Device pointer to right-hand side (modified to contain solution)
/// * `ldb` - Leading dimension of B
/// * `info` - Device pointer to info value
#[inline]
pub fn posv<T: PosvType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    B: *mut T,
    ldb: i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe { T::posv(handle.as_raw(), uplo.into(), n, nrhs, A, lda, B, ldb, info) };
    Error::from_status(status)
}

/// Batched version of posv.
#[inline]
pub fn posv_batched<T: PosvType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    nrhs: i32,
    A: *const *mut T,
    lda: i32,
    B: *const *mut T,
    ldb: i32,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::posv_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of posv.
#[inline]
pub fn posv_strided_batched<T: PosvType>(
    handle: &Handle,
    uplo: Fill,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    B: *mut T,
    ldb: i32,
    stride_b: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::posv_strided_batched(
            handle.as_raw(),
            uplo.into(),
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Solves overdetermined or underdetermined linear systems using QR/LQ.
///
/// - If m >= n: solves the least squares problem min ||B - A*X||
/// - If m < n: solves the minimum norm problem min ||X|| subject to A*X = B
///
/// # Arguments
/// * `handle` - rocBLAS handle
/// * `trans` - Specifies whether to use A or A^T/A^H
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `nrhs` - Number of right-hand sides
/// * `A` - Device pointer to m-by-n matrix (modified)
/// * `lda` - Leading dimension of A
/// * `B` - Device pointer to right-hand side (modified to contain solution)
/// * `ldb` - Leading dimension of B
/// * `info` - Device pointer to info value
#[inline]
pub fn gels<T: GelsType>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    B: *mut T,
    ldb: i32,
    info: *mut i32,
) -> Result<()> {
    let status = unsafe {
        T::gels(
            handle.as_raw(),
            trans.into(),
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
        )
    };
    Error::from_status(status)
}

/// Batched version of gels.
#[inline]
pub fn gels_batched<T: GelsType>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    nrhs: i32,
    A: *const *mut T,
    lda: i32,
    B: *const *mut T,
    ldb: i32,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gels_batched(
            handle.as_raw(),
            trans.into(),
            m,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}

/// Strided batched version of gels.
#[inline]
pub fn gels_strided_batched<T: GelsType>(
    handle: &Handle,
    trans: Operation,
    m: i32,
    n: i32,
    nrhs: i32,
    A: *mut T,
    lda: i32,
    stride_a: i64,
    B: *mut T,
    ldb: i32,
    stride_b: i64,
    info: *mut i32,
    batch_count: i32,
) -> Result<()> {
    let status = unsafe {
        T::gels_strided_batched(
            handle.as_raw(),
            trans.into(),
            m,
            n,
            nrhs,
            A,
            lda,
            stride_a,
            B,
            ldb,
            stride_b,
            info,
            batch_count,
        )
    };
    Error::from_status(status)
}
