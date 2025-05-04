// src/rocsolver/types.rs

use crate::rocblas::ffi;

/// Direction for Householder transforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direct {
    /// Forward direction
    Forward,
    /// Backward direction
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
            _ => Direct::Forward, // Default for unknown values
        }
    }
}

/// Storage format for Householder vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storev {
    /// Column-wise storage
    ColumnWise,
    /// Row-wise storage
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
            _ => Storev::ColumnWise, // Default for unknown values
        }
    }
}

/// Specifies how singular vectors should be computed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Svect {
    /// Compute all singular vectors
    All,
    /// Compute only singular vectors
    Singular,
    /// Overwrite input with singular vectors
    Overwrite,
    /// Don't compute singular vectors
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
            _ => Svect::None, // Default for unknown values
        }
    }
}

/// Work mode for computations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workmode {
    /// Out-of-place computation
    OutOfPlace,
    /// In-place computation
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
            _ => Workmode::OutOfPlace, // Default for unknown values
        }
    }
}

/// Specifies how eigenvectors should be computed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evect {
    /// Compute eigenvectors for original matrix
    Original,
    /// Compute eigenvectors for tridiagonal matrix
    Tridiagonal,
    /// Don't compute eigenvectors
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
            _ => Evect::None, // Default for unknown values
        }
    }
}

/// Form of generalized eigenproblem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eform {
    /// A*x = lambda*B*x
    Ax,
    /// A*B*x = lambda*x
    ABx,
    /// B*A*x = lambda*x
    BAx,
}

impl From<Eform> for ffi::rocblas_eform {
    fn from(eform: Eform) -> Self {
        match eform {
            Eform::Ax => ffi::rocblas_eform__rocblas_eform_ax,
            Eform::ABx => ffi::rocblas_eform__rocblas_eform_abx,
            Eform::BAx => ffi::rocblas_eform__rocblas_eform_bax,
        }
    }
}

impl From<ffi::rocblas_eform> for Eform {
    fn from(eform: ffi::rocblas_eform) -> Self {
        match eform {
            ffi::rocblas_eform__rocblas_eform_ax => Eform::Ax,
            ffi::rocblas_eform__rocblas_eform_abx => Eform::ABx,
            ffi::rocblas_eform__rocblas_eform_bax => Eform::BAx,
            _ => Eform::Ax, // Default for unknown values
        }
    }
}

/// Range type for eigenvalues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Erange {
    /// All eigenvalues
    All,
    /// Eigenvalues in a value range
    Value,
    /// Eigenvalues in an index range
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
            _ => Erange::All, // Default for unknown values
        }
    }
}

/// Eigenvalue ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eorder {
    /// Group eigenvalues by blocks
    Blocks,
    /// Order eigenvalues for entire matrix
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
            _ => Eorder::Blocks, // Default for unknown values
        }
    }
}

/// Eigenvalue sorting order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Esort {
    /// No sorting
    None,
    /// Sort in ascending order
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
            _ => Esort::None, // Default for unknown values
        }
    }
}

/// Range type for singular values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Srange {
    /// All singular values
    All,
    /// Singular values in a value range
    Value,
    /// Singular values in an index range
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
            _ => Srange::All, // Default for unknown values
        }
    }
}

/// Opaque wrapper for rfinfo handle
pub struct RfInfo {
    handle: ffi::rocsolver_rfinfo,
}

impl RfInfo {
    /// Get the raw handle
    pub fn as_raw(&self) -> ffi::rocsolver_rfinfo {
        self.handle
    }
}

/// Mode for rfinfo
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfinfoMode {
    /// LU factorization
    LU,
    /// Cholesky factorization
    Cholesky,
}

impl From<RfinfoMode> for ffi::rocsolver_rfinfo_mode {
    fn from(mode: RfinfoMode) -> Self {
        match mode {
            RfinfoMode::LU => ffi::rocsolver_rfinfo_mode__rocsolver_rfinfo_mode_lu,
            RfinfoMode::Cholesky => ffi::rocsolver_rfinfo_mode__rocsolver_rfinfo_mode_cholesky,
        }
    }
}

impl From<ffi::rocsolver_rfinfo_mode> for RfinfoMode {
    fn from(mode: ffi::rocsolver_rfinfo_mode) -> Self {
        match mode {
            ffi::rocsolver_rfinfo_mode__rocsolver_rfinfo_mode_lu => RfinfoMode::LU,
            ffi::rocsolver_rfinfo_mode__rocsolver_rfinfo_mode_cholesky => RfinfoMode::Cholesky,
            _ => RfinfoMode::LU, // Default for unknown values
        }
    }
}