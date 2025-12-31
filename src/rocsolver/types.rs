// src/rocsolver/types.rs
//! Type-safe enums for rocSOLVER operations.
//!
//! This module provides Rust-idiomatic enums for the various options and modes
//! used in rocSOLVER/LAPACK operations.

use crate::rocsolver::ffi;

// Re-export common types from rocBLAS that are shared
pub use crate::rocblas::types::{DataType, Diagonal, Fill, Operation, Side};

/// Specifies how to store/compute vectors in SVD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Svect {
    /// Compute all singular vectors.
    All,
    /// Compute only singular vectors (thin SVD).
    Singular,
    /// Overwrite input matrix with singular vectors.
    Overwrite,
    /// Do not compute singular vectors.
    None,
}

impl From<Svect> for ffi::rocblas_svect {
    fn from(svect: Svect) -> Self {
        match svect {
            Svect::All => ffi::rocblas_svect__rocblas_svect_all,
            Svect::Singular => ffi::rocblas_svect__rocblas_svect_singular,
            Svect::Overwrite => ffi::rocblas_svect__rocblas_svect_overwrite,
            Svect::None => ffi::rocblas_svect__rocblas_svect_none,
        }
    }
}

impl From<ffi::rocblas_svect> for Svect {
    fn from(svect: ffi::rocblas_svect) -> Self {
        match svect {
            ffi::rocblas_svect__rocblas_svect_all => Svect::All,
            ffi::rocblas_svect__rocblas_svect_singular => Svect::Singular,
            ffi::rocblas_svect__rocblas_svect_overwrite => Svect::Overwrite,
            ffi::rocblas_svect__rocblas_svect_none => Svect::None,
            _ => Svect::None,
        }
    }
}

/// Specifies how to compute eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Evect {
    /// Compute eigenvectors of the original matrix.
    Original,
    /// Compute eigenvectors of the tridiagonal matrix.
    Tridiagonal,
    /// Do not compute eigenvectors.
    None,
}

impl From<Evect> for ffi::rocblas_evect {
    fn from(evect: Evect) -> Self {
        match evect {
            Evect::Original => ffi::rocblas_evect__rocblas_evect_original,
            Evect::Tridiagonal => ffi::rocblas_evect__rocblas_evect_tridiagonal,
            Evect::None => ffi::rocblas_evect__rocblas_evect_none,
        }
    }
}

impl From<ffi::rocblas_evect> for Evect {
    fn from(evect: ffi::rocblas_evect) -> Self {
        match evect {
            ffi::rocblas_evect__rocblas_evect_original => Evect::Original,
            ffi::rocblas_evect__rocblas_evect_tridiagonal => Evect::Tridiagonal,
            ffi::rocblas_evect__rocblas_evect_none => Evect::None,
            _ => Evect::None,
        }
    }
}

/// Specifies the range of eigenvalues to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Erange {
    /// Compute all eigenvalues.
    All,
    /// Compute eigenvalues in a half-open interval (vl, vu].
    Value,
    /// Compute the il-th through iu-th eigenvalues.
    Index,
}

impl From<Erange> for ffi::rocblas_erange {
    fn from(erange: Erange) -> Self {
        match erange {
            Erange::All => ffi::rocblas_erange__rocblas_erange_all,
            Erange::Value => ffi::rocblas_erange__rocblas_erange_value,
            Erange::Index => ffi::rocblas_erange__rocblas_erange_index,
        }
    }
}

impl From<ffi::rocblas_erange> for Erange {
    fn from(erange: ffi::rocblas_erange) -> Self {
        match erange {
            ffi::rocblas_erange__rocblas_erange_all => Erange::All,
            ffi::rocblas_erange__rocblas_erange_value => Erange::Value,
            ffi::rocblas_erange__rocblas_erange_index => Erange::Index,
            _ => Erange::All,
        }
    }
}

/// Specifies how vectors are stored in transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storev {
    /// Vectors stored column-wise.
    ColumnWise,
    /// Vectors stored row-wise.
    RowWise,
}

impl From<Storev> for ffi::rocblas_storev {
    fn from(storev: Storev) -> Self {
        match storev {
            Storev::ColumnWise => ffi::rocblas_storev__rocblas_column_wise,
            Storev::RowWise => ffi::rocblas_storev__rocblas_row_wise,
        }
    }
}

impl From<ffi::rocblas_storev> for Storev {
    fn from(storev: ffi::rocblas_storev) -> Self {
        match storev {
            ffi::rocblas_storev__rocblas_column_wise => Storev::ColumnWise,
            ffi::rocblas_storev__rocblas_row_wise => Storev::RowWise,
            _ => Storev::ColumnWise,
        }
    }
}

/// Specifies the direction of Householder application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direct {
    /// Apply in forward direction.
    Forward,
    /// Apply in backward direction.
    Backward,
}

impl From<Direct> for ffi::rocblas_direct {
    fn from(direct: Direct) -> Self {
        match direct {
            Direct::Forward => ffi::rocblas_direct__rocblas_forward_direction,
            Direct::Backward => ffi::rocblas_direct__rocblas_backward_direction,
        }
    }
}

impl From<ffi::rocblas_direct> for Direct {
    fn from(direct: ffi::rocblas_direct) -> Self {
        match direct {
            ffi::rocblas_direct__rocblas_forward_direction => Direct::Forward,
            ffi::rocblas_direct__rocblas_backward_direction => Direct::Backward,
            _ => Direct::Forward,
        }
    }
}

/// Specifies whether to use in-place or out-of-place algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Workmode {
    /// Use out-of-place algorithm (requires extra workspace).
    OutOfPlace,
    /// Use in-place algorithm (slower but less memory).
    InPlace,
}

impl From<Workmode> for ffi::rocblas_workmode {
    fn from(workmode: Workmode) -> Self {
        match workmode {
            Workmode::OutOfPlace => ffi::rocblas_workmode__rocblas_outofplace,
            Workmode::InPlace => ffi::rocblas_workmode__rocblas_inplace,
        }
    }
}

impl From<ffi::rocblas_workmode> for Workmode {
    fn from(workmode: ffi::rocblas_workmode) -> Self {
        match workmode {
            ffi::rocblas_workmode__rocblas_outofplace => Workmode::OutOfPlace,
            ffi::rocblas_workmode__rocblas_inplace => Workmode::InPlace,
            _ => Workmode::OutOfPlace,
        }
    }
}

/// Specifies the form of the generalized eigenvalue problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Eform {
    /// A*x = lambda*B*x
    Ax,
    /// A*B*x = lambda*x
    Abx,
    /// B*A*x = lambda*x
    Bax,
}

impl From<Eform> for ffi::rocblas_eform {
    fn from(eform: Eform) -> Self {
        match eform {
            Eform::Ax => ffi::rocblas_eform__rocblas_eform_ax,
            Eform::Abx => ffi::rocblas_eform__rocblas_eform_abx,
            Eform::Bax => ffi::rocblas_eform__rocblas_eform_bax,
        }
    }
}

impl From<ffi::rocblas_eform> for Eform {
    fn from(eform: ffi::rocblas_eform) -> Self {
        match eform {
            ffi::rocblas_eform__rocblas_eform_ax => Eform::Ax,
            ffi::rocblas_eform__rocblas_eform_abx => Eform::Abx,
            ffi::rocblas_eform__rocblas_eform_bax => Eform::Bax,
            _ => Eform::Ax,
        }
    }
}

/// Specifies the order of eigenvalue/eigenvector computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Eorder {
    /// Compute eigenvalues in blocks.
    Blocks,
    /// Compute all eigenvalues at once.
    Entire,
}

impl From<Eorder> for ffi::rocblas_eorder {
    fn from(eorder: Eorder) -> Self {
        match eorder {
            Eorder::Blocks => ffi::rocblas_eorder__rocblas_eorder_blocks,
            Eorder::Entire => ffi::rocblas_eorder__rocblas_eorder_entire,
        }
    }
}

impl From<ffi::rocblas_eorder> for Eorder {
    fn from(eorder: ffi::rocblas_eorder) -> Self {
        match eorder {
            ffi::rocblas_eorder__rocblas_eorder_blocks => Eorder::Blocks,
            ffi::rocblas_eorder__rocblas_eorder_entire => Eorder::Entire,
            _ => Eorder::Entire,
        }
    }
}

/// Specifies how eigenvalues should be sorted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Esort {
    /// No sorting.
    None,
    /// Sort in ascending order.
    Ascending,
}

impl From<Esort> for ffi::rocblas_esort {
    fn from(esort: Esort) -> Self {
        match esort {
            Esort::None => ffi::rocblas_esort__rocblas_esort_none,
            Esort::Ascending => ffi::rocblas_esort__rocblas_esort_ascending,
        }
    }
}

impl From<ffi::rocblas_esort> for Esort {
    fn from(esort: ffi::rocblas_esort) -> Self {
        match esort {
            ffi::rocblas_esort__rocblas_esort_none => Esort::None,
            ffi::rocblas_esort__rocblas_esort_ascending => Esort::Ascending,
            _ => Esort::None,
        }
    }
}

/// Specifies the algorithm mode for rocSOLVER functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgMode {
    /// Use GPU-only algorithm.
    Gpu,
    /// Use hybrid CPU-GPU algorithm.
    Hybrid,
    /// Use mixed precision algorithm.
    Mixed,
}

impl From<AlgMode> for ffi::rocsolver_alg_mode {
    fn from(mode: AlgMode) -> Self {
        match mode {
            AlgMode::Gpu => ffi::rocsolver_alg_mode__rocsolver_alg_mode_gpu,
            AlgMode::Hybrid => ffi::rocsolver_alg_mode__rocsolver_alg_mode_hybrid,
            AlgMode::Mixed => ffi::rocsolver_alg_mode__rocsolver_alg_mode_mixed,
        }
    }
}

impl From<ffi::rocsolver_alg_mode> for AlgMode {
    fn from(mode: ffi::rocsolver_alg_mode) -> Self {
        match mode {
            ffi::rocsolver_alg_mode__rocsolver_alg_mode_gpu => AlgMode::Gpu,
            ffi::rocsolver_alg_mode__rocsolver_alg_mode_hybrid => AlgMode::Hybrid,
            ffi::rocsolver_alg_mode__rocsolver_alg_mode_mixed => AlgMode::Mixed,
            _ => AlgMode::Gpu,
        }
    }
}

/// Specifies the singular value range for partial SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Srange {
    /// Compute all singular values.
    All,
    /// Compute singular values in a half-open interval (vl, vu].
    Value,
    /// Compute the il-th through iu-th singular values.
    Index,
}

impl From<Srange> for ffi::rocblas_srange {
    fn from(srange: Srange) -> Self {
        match srange {
            Srange::All => ffi::rocblas_srange__rocblas_srange_all,
            Srange::Value => ffi::rocblas_srange__rocblas_srange_value,
            Srange::Index => ffi::rocblas_srange__rocblas_srange_index,
        }
    }
}

impl From<ffi::rocblas_srange> for Srange {
    fn from(srange: ffi::rocblas_srange) -> Self {
        match srange {
            ffi::rocblas_srange__rocblas_srange_all => Srange::All,
            ffi::rocblas_srange__rocblas_srange_value => Srange::Value,
            ffi::rocblas_srange__rocblas_srange_index => Srange::Index,
            _ => Srange::All,
        }
    }
}

/// Complex number type for single precision (f32).
///
/// This uses the rocsolver bindings type to ensure compatibility with rocsolver functions.
pub type Complex32 = crate::rocsolver::bindings::rocblas_float_complex;

/// Complex number type for double precision (f64).
///
/// This uses the rocsolver bindings type to ensure compatibility with rocsolver functions.
pub type Complex64 = crate::rocsolver::bindings::rocblas_double_complex;

impl Complex32 {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { x: re, y: im }
    }

    /// Get the real part.
    #[inline]
    pub fn re(&self) -> f32 {
        self.x
    }

    /// Get the imaginary part.
    #[inline]
    pub fn im(&self) -> f32 {
        self.y
    }
}

impl Complex64 {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { x: re, y: im }
    }

    /// Get the real part.
    #[inline]
    pub fn re(&self) -> f64 {
        self.x
    }

    /// Get the imaginary part.
    #[inline]
    pub fn im(&self) -> f64 {
        self.y
    }
}
