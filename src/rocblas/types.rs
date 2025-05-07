// src/rocblas/types.rs

use crate::rocblas::ffi;

// Re-export the basic types
pub use ffi::rocblas_bfloat16;
pub use ffi::rocblas_double_complex;
pub use ffi::rocblas_float_complex;
pub use ffi::rocblas_half;

/// Enum for matrix operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Operate with the matrix
    None,
    /// Operate with the transpose of the matrix
    Transpose,
    /// Operate with the conjugate transpose of the matrix
    ConjugateTranspose,
}

impl From<Operation> for ffi::rocblas_operation {
    fn from(op: Operation) -> Self {
        match op {
            Operation::None => ffi::rocblas_operation__rocblas_operation_none,
            Operation::Transpose => ffi::rocblas_operation__rocblas_operation_transpose,
            Operation::ConjugateTranspose => {
                ffi::rocblas_operation__rocblas_operation_conjugate_transpose
            }
        }
    }
}

impl From<ffi::rocblas_operation> for Operation {
    fn from(op: ffi::rocblas_operation) -> Self {
        match op {
            ffi::rocblas_operation__rocblas_operation_none => Operation::None,
            ffi::rocblas_operation__rocblas_operation_transpose => Operation::Transpose,
            ffi::rocblas_operation__rocblas_operation_conjugate_transpose => {
                Operation::ConjugateTranspose
            }
            _ => Operation::None, // Default to None for unknown values
        }
    }
}

/// Enum for matrix fill modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// Upper triangle
    Upper,
    /// Lower triangle
    Lower,
    /// Full matrix
    Full,
}

impl From<Fill> for ffi::rocblas_fill {
    fn from(fill: Fill) -> Self {
        match fill {
            Fill::Upper => ffi::rocblas_fill__rocblas_fill_upper,
            Fill::Lower => ffi::rocblas_fill__rocblas_fill_lower,
            Fill::Full => ffi::rocblas_fill__rocblas_fill_full,
        }
    }
}

impl From<ffi::rocblas_fill> for Fill {
    fn from(fill: ffi::rocblas_fill) -> Self {
        match fill {
            ffi::rocblas_fill__rocblas_fill_upper => Fill::Upper,
            ffi::rocblas_fill__rocblas_fill_lower => Fill::Lower,
            ffi::rocblas_fill__rocblas_fill_full => Fill::Full,
            _ => Fill::Full, // Default to Full for unknown values
        }
    }
}

/// Enum for diagonal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagonal {
    /// Non-unit triangular
    NonUnit,
    /// Unit triangular
    Unit,
}

impl From<Diagonal> for ffi::rocblas_diagonal {
    fn from(diag: Diagonal) -> Self {
        match diag {
            Diagonal::NonUnit => ffi::rocblas_diagonal__rocblas_diagonal_non_unit,
            Diagonal::Unit => ffi::rocblas_diagonal__rocblas_diagonal_unit,
        }
    }
}

impl From<ffi::rocblas_diagonal> for Diagonal {
    fn from(diag: ffi::rocblas_diagonal) -> Self {
        match diag {
            ffi::rocblas_diagonal__rocblas_diagonal_non_unit => Diagonal::NonUnit,
            ffi::rocblas_diagonal__rocblas_diagonal_unit => Diagonal::Unit,
            _ => Diagonal::NonUnit, // Default to NonUnit for unknown values
        }
    }
}

/// Enum for matrix side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Multiply general matrix by symmetric, Hermitian, or triangular matrix on the left
    Left,
    /// Multiply general matrix by symmetric, Hermitian, or triangular matrix on the right
    Right,
    /// Multiply on both sides
    Both,
}

impl From<Side> for ffi::rocblas_side {
    fn from(side: Side) -> Self {
        match side {
            Side::Left => ffi::rocblas_side__rocblas_side_left,
            Side::Right => ffi::rocblas_side__rocblas_side_right,
            Side::Both => ffi::rocblas_side__rocblas_side_both,
        }
    }
}

impl From<ffi::rocblas_side> for Side {
    fn from(side: ffi::rocblas_side) -> Self {
        match side {
            ffi::rocblas_side__rocblas_side_left => Side::Left,
            ffi::rocblas_side__rocblas_side_right => Side::Right,
            ffi::rocblas_side__rocblas_side_both => Side::Both,
            _ => Side::Left, // Default to Left for unknown values
        }
    }
}

/// Enum for data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// 16-bit floating point, real
    F16Real,
    /// 32-bit floating point, real
    F32Real,
    /// 64-bit floating point, real
    F64Real,
    /// 16-bit floating point, complex
    F16Complex,
    /// 32-bit floating point, complex
    F32Complex,
    /// 64-bit floating point, complex
    F64Complex,
    /// 8-bit signed integer, real
    I8Real,
    /// 8-bit unsigned integer, real
    U8Real,
    /// 32-bit signed integer, real
    I32Real,
    /// 32-bit unsigned integer, real
    U32Real,
    /// 8-bit signed integer, complex
    I8Complex,
    /// 8-bit unsigned integer, complex
    U8Complex,
    /// 32-bit signed integer, complex
    I32Complex,
    /// 32-bit unsigned integer, complex
    U32Complex,
    /// 16-bit bfloat, real
    BF16Real,
    /// 16-bit bfloat, complex
    BF16Complex,
    /// 8-bit floating point, real
    F8Real,
    /// 8-bit bfloat, real
    BF8Real,
}

impl From<DataType> for ffi::rocblas_datatype {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::F16Real => ffi::rocblas_datatype__rocblas_datatype_f16_r,
            DataType::F32Real => ffi::rocblas_datatype__rocblas_datatype_f32_r,
            DataType::F64Real => ffi::rocblas_datatype__rocblas_datatype_f64_r,
            DataType::F16Complex => ffi::rocblas_datatype__rocblas_datatype_f16_c,
            DataType::F32Complex => ffi::rocblas_datatype__rocblas_datatype_f32_c,
            DataType::F64Complex => ffi::rocblas_datatype__rocblas_datatype_f64_c,
            DataType::I8Real => ffi::rocblas_datatype__rocblas_datatype_i8_r,
            DataType::U8Real => ffi::rocblas_datatype__rocblas_datatype_u8_r,
            DataType::I32Real => ffi::rocblas_datatype__rocblas_datatype_i32_r,
            DataType::U32Real => ffi::rocblas_datatype__rocblas_datatype_u32_r,
            DataType::I8Complex => ffi::rocblas_datatype__rocblas_datatype_i8_c,
            DataType::U8Complex => ffi::rocblas_datatype__rocblas_datatype_u8_c,
            DataType::I32Complex => ffi::rocblas_datatype__rocblas_datatype_i32_c,
            DataType::U32Complex => ffi::rocblas_datatype__rocblas_datatype_u32_c,
            DataType::BF16Real => ffi::rocblas_datatype__rocblas_datatype_bf16_r,
            DataType::BF16Complex => ffi::rocblas_datatype__rocblas_datatype_bf16_c,
            DataType::F8Real => ffi::rocblas_datatype__rocblas_datatype_f8_r,
            DataType::BF8Real => ffi::rocblas_datatype__rocblas_datatype_bf8_r,
        }
    }
}

impl From<ffi::rocblas_datatype> for DataType {
    fn from(dtype: ffi::rocblas_datatype) -> Self {
        match dtype {
            ffi::rocblas_datatype__rocblas_datatype_f16_r => DataType::F16Real,
            ffi::rocblas_datatype__rocblas_datatype_f32_r => DataType::F32Real,
            ffi::rocblas_datatype__rocblas_datatype_f64_r => DataType::F64Real,
            ffi::rocblas_datatype__rocblas_datatype_f16_c => DataType::F16Complex,
            ffi::rocblas_datatype__rocblas_datatype_f32_c => DataType::F32Complex,
            ffi::rocblas_datatype__rocblas_datatype_f64_c => DataType::F64Complex,
            ffi::rocblas_datatype__rocblas_datatype_i8_r => DataType::I8Real,
            ffi::rocblas_datatype__rocblas_datatype_u8_r => DataType::U8Real,
            ffi::rocblas_datatype__rocblas_datatype_i32_r => DataType::I32Real,
            ffi::rocblas_datatype__rocblas_datatype_u32_r => DataType::U32Real,
            ffi::rocblas_datatype__rocblas_datatype_i8_c => DataType::I8Complex,
            ffi::rocblas_datatype__rocblas_datatype_u8_c => DataType::U8Complex,
            ffi::rocblas_datatype__rocblas_datatype_i32_c => DataType::I32Complex,
            ffi::rocblas_datatype__rocblas_datatype_u32_c => DataType::U32Complex,
            ffi::rocblas_datatype__rocblas_datatype_bf16_r => DataType::BF16Real,
            ffi::rocblas_datatype__rocblas_datatype_bf16_c => DataType::BF16Complex,
            ffi::rocblas_datatype__rocblas_datatype_f8_r => DataType::F8Real,
            ffi::rocblas_datatype__rocblas_datatype_bf8_r => DataType::BF8Real,
            _ => DataType::F32Real, // Default to F32Real for unknown values
        }
    }
}

// Re-export the types with their rocblas_ prefixes for compatibility
pub use ffi::rocblas_datatype;
pub use ffi::rocblas_diagonal;
pub use ffi::rocblas_fill;
pub use ffi::rocblas_operation;
pub use ffi::rocblas_side;
