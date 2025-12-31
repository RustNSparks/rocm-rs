// src/rocsolver/lapack/mod.rs
//! LAPACK-style linear algebra operations.
//!
//! This module provides safe wrappers for rocSOLVER's LAPACK functions, organized
//! by mathematical category:
//!
//! - [`decompositions`] - Matrix factorizations (QR, LU, Cholesky, Bidiagonal)
//! - [`solvers`] - Linear system solvers
//! - [`svd`] - Singular Value Decomposition
//! - [`eigenvalue`] - Eigenvalue computations
//! - [`orthogonal`] - Orthogonal/Unitary matrix operations

pub mod decompositions;
pub mod eigenvalue;
pub mod orthogonal;
pub mod solvers;
pub mod svd;

// Re-export commonly used functions at the lapack module level
pub use decompositions::{
    gebrd, gebrd_batched, gebrd_strided_batched, geqrf, geqrf_batched, geqrf_strided_batched,
    getrf, getrf_batched, getrf_npvt, getrf_npvt_batched, getrf_npvt_strided_batched,
    getrf_strided_batched, potrf, potrf_batched, potrf_strided_batched,
};

pub use solvers::{
    gels, gels_batched, gels_strided_batched, gesv, gesv_batched, gesv_strided_batched, getrs,
    getrs_batched, getrs_strided_batched, posv, posv_batched, posv_strided_batched,
};

pub use svd::gesvd;

pub use eigenvalue::{
    heev, heev_batched, heev_strided_batched, syev, syev_batched, syev_strided_batched,
};

pub use orthogonal::{orgqr, ormqr, ungqr, unmqr};
