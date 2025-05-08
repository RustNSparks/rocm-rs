//! Matrix descriptor types and enums

use std::mem::MaybeUninit;
use crate::rocsparse::{rocsparse_create_mat_descr, rocsparse_destroy_mat_descr, rocsparse_direction_, rocsparse_direction__rocsparse_direction_column, rocsparse_direction__rocsparse_direction_row, rocsparse_get_mat_index_base, rocsparse_get_mat_type, rocsparse_index_base_, rocsparse_index_base__rocsparse_index_base_one, rocsparse_index_base__rocsparse_index_base_zero, rocsparse_mat_descr, rocsparse_matrix_type_, rocsparse_matrix_type__rocsparse_matrix_type_general, rocsparse_matrix_type__rocsparse_matrix_type_hermitian, rocsparse_matrix_type__rocsparse_matrix_type_symmetric, rocsparse_matrix_type__rocsparse_matrix_type_triangular, rocsparse_set_mat_index_base, rocsparse_set_mat_type};
use crate::rocsparse::error::*;

/// Matrix storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixType {
    /// General matrix
    General,
    /// Symmetric matrix
    Symmetric,
    /// Hermitian matrix
    Hermitian,
    /// Triangular matrix
    Triangular,
}

impl From<MatrixType> for rocsparse_matrix_type_ {
    fn from(ty: MatrixType) -> Self {
        match ty {
            MatrixType::General => rocsparse_matrix_type__rocsparse_matrix_type_general,
            MatrixType::Symmetric => rocsparse_matrix_type__rocsparse_matrix_type_symmetric,
            MatrixType::Hermitian => rocsparse_matrix_type__rocsparse_matrix_type_hermitian,
            MatrixType::Triangular => rocsparse_matrix_type__rocsparse_matrix_type_triangular,
        }
    }
}

/// Index base for sparse matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBase {
    /// Zero-based indexing
    Zero,
    /// One-based indexing
    One,
}

impl From<IndexBase> for rocsparse_index_base_ {
    fn from(base: IndexBase) -> Self {
        match base {
            IndexBase::Zero => rocsparse_index_base__rocsparse_index_base_zero,
            IndexBase::One => rocsparse_index_base__rocsparse_index_base_one,
        }
    }
}

/// Direction for block storage formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Parse matrix by rows
    Row,
    /// Parse matrix by columns
    Column,
}

impl From<Direction> for rocsparse_direction_ {
    fn from(dir: Direction) -> Self {
        match dir {
            Direction::Row => rocsparse_direction__rocsparse_direction_row,
            Direction::Column => rocsparse_direction__rocsparse_direction_column,
        }
    }
}

/// Matrix descriptor for sparse matrices
pub struct MatrixDescriptor {
    pub(crate) inner: rocsparse_mat_descr,
}

impl MatrixDescriptor {
    /// Create a new matrix descriptor
    pub fn new() -> Result<Self> {
        let mut descr = MaybeUninit::uninit();
        let status = unsafe { rocsparse_create_mat_descr(descr.as_mut_ptr()) };
        status_to_result(status)?;
        let descr = unsafe { descr.assume_init() };
        Ok(Self { inner: descr })
    }

    /// Set the index base
    pub fn set_index_base(&self, base: IndexBase) -> Result<()> {
        let status = unsafe { rocsparse_set_mat_index_base(self.inner, base.into()) };
        status_to_result(status)
    }

    /// Get the index base
    pub fn get_index_base(&self) -> IndexBase {
        let base = unsafe { rocsparse_get_mat_index_base(self.inner) };
        if base == rocsparse_index_base__rocsparse_index_base_one {
            IndexBase::One
        } else {
            IndexBase::Zero
        }
    }

    /// Set the matrix type
    pub fn set_matrix_type(&self, ty: MatrixType) -> Result<()> {
        let status = unsafe { rocsparse_set_mat_type(self.inner, ty.into()) };
        status_to_result(status)
    }

    /// Get the matrix type
    pub fn get_matrix_type(&self) -> MatrixType {
        let ty = unsafe { rocsparse_get_mat_type(self.inner) };
        match ty {
            rocsparse_matrix_type__rocsparse_matrix_type_symmetric => MatrixType::Symmetric,
            rocsparse_matrix_type__rocsparse_matrix_type_hermitian => MatrixType::Hermitian,
            rocsparse_matrix_type__rocsparse_matrix_type_triangular => MatrixType::Triangular,
            _ => MatrixType::General,
        }
    }
}

impl Drop for MatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            // Ignore error on drop
            let _ = rocsparse_destroy_mat_descr(self.inner);
        }
    }
}