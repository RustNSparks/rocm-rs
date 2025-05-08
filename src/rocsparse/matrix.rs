//! Sparse matrix types and formats

use std::mem::MaybeUninit;
use std::marker::PhantomData;
use crate::rocsparse::error::{Result, status_to_result};
use crate::rocsparse::descriptor::IndexBase;
use crate::rocsparse::{rocsparse_create_hyb_mat, rocsparse_create_mat_info, rocsparse_destroy_hyb_mat, rocsparse_destroy_mat_info, rocsparse_destroy_spmat_descr, rocsparse_hyb_mat, rocsparse_hyb_partition_, rocsparse_hyb_partition__rocsparse_hyb_partition_auto, rocsparse_hyb_partition__rocsparse_hyb_partition_max, rocsparse_hyb_partition__rocsparse_hyb_partition_user, rocsparse_mat_info, rocsparse_spmat_descr};

/// HYB matrix partitioning type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybPartition {
    /// Automatically decide on ELL nnz per row
    Auto,
    /// User given ELL nnz per row
    User,
    /// Max ELL nnz per row, no COO part
    Max,
}

impl From<HybPartition> for rocsparse_hyb_partition_ {
    fn from(partition: HybPartition) -> Self {
        match partition {
            HybPartition::Auto => rocsparse_hyb_partition__rocsparse_hyb_partition_auto,
            HybPartition::User => rocsparse_hyb_partition__rocsparse_hyb_partition_user,
            HybPartition::Max => rocsparse_hyb_partition__rocsparse_hyb_partition_max,
        }
    }
}

/// Hybrid matrix format (ELL + COO)
pub struct HybMatrix {
    pub(crate) inner: rocsparse_hyb_mat,
}

impl HybMatrix {
    /// Create a new HYB matrix
    pub fn new() -> Result<Self> {
        let mut hyb = MaybeUninit::uninit();
        let status = unsafe { rocsparse_create_hyb_mat(hyb.as_mut_ptr()) };
        status_to_result(status)?;
        let hyb = unsafe { hyb.assume_init() };
        Ok(Self { inner: hyb })
    }
}

impl Drop for HybMatrix {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_hyb_mat(self.inner);
        }
    }
}

/// Matrix info structure
pub struct MatrixInfo {
    pub(crate) inner: rocsparse_mat_info,
}

impl MatrixInfo {
    /// Create a new matrix info
    pub fn new() -> Result<Self> {
        let mut info = MaybeUninit::uninit();
        let status = unsafe { rocsparse_create_mat_info(info.as_mut_ptr()) };
        status_to_result(status)?;
        let info = unsafe { info.assume_init() };
        Ok(Self { inner: info })
    }
}

impl Drop for MatrixInfo {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_mat_info(self.inner);
        }
    }
}

/// Sparse matrix representation
pub struct SparseMatrix<T> {
    pub(crate) inner: rocsparse_spmat_descr,
    _phantom: PhantomData<T>,
}

impl<T> Drop for SparseMatrix<T> {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_spmat_descr(self.inner);
        }
    }
}

/// CSR (Compressed Sparse Row) matrix format helper
pub struct CsrMatrix<T> {
    /// Number of rows
    pub rows: i32,
    /// Number of columns
    pub cols: i32,
    /// Row pointers
    pub row_ptr: Vec<i32>,
    /// Column indices
    pub col_ind: Vec<i32>,
    /// Values
    pub values: Vec<T>,
    /// Index base (zero or one)
    pub index_base: IndexBase
}