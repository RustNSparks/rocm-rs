/*!
# rocBLAS Type Definitions

This module provides Rust enums and type aliases for the rocBLAS C types.
*/

use crate::rocblas::bindings;

/// Operation type for matrix operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Operation {
    /// No operation, use matrix as-is
    None,
    /// Transpose the matrix
    Transpose,
    /// Conjugate transpose (transpose and complex conjugate)
    ConjugateTranspose,
}

impl From<Operation> for u32 {
    fn from(op: Operation) -> Self {
        match op {
            Operation::None => bindings::rocblas_operation__rocblas_operation_none,
            Operation::Transpose => bindings::rocblas_operation__rocblas_operation_transpose,
            Operation::ConjugateTranspose => {
                bindings::rocblas_operation__rocblas_operation_conjugate_transpose
            }
        }
    }
}

impl From<u32> for Operation {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_operation__rocblas_operation_none => Operation::None,
            bindings::rocblas_operation__rocblas_operation_transpose => Operation::Transpose,
            bindings::rocblas_operation__rocblas_operation_conjugate_transpose => {
                Operation::ConjugateTranspose
            }
            _ => panic!("Invalid rocblas_operation value"),
        }
    }
}

/// Triangle part for matrix operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Fill {
    /// Use upper triangle
    Upper,
    /// Use lower triangle
    Lower,
    /// Use full matrix
    Full,
}

impl From<Fill> for u32 {
    fn from(fill: Fill) -> Self {
        match fill {
            Fill::Upper => bindings::rocblas_fill__rocblas_fill_upper,
            Fill::Lower => bindings::rocblas_fill__rocblas_fill_lower,
            Fill::Full => bindings::rocblas_fill__rocblas_fill_full,
        }
    }
}

impl From<u32> for Fill {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_fill__rocblas_fill_upper => Fill::Upper,
            bindings::rocblas_fill__rocblas_fill_lower => Fill::Lower,
            bindings::rocblas_fill__rocblas_fill_full => Fill::Full,
            _ => panic!("Invalid rocblas_fill value"),
        }
    }
}

/// Diagonal unit property for triangular matrices
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Diagonal {
    /// Non-unit diagonal (use the values on the diagonal)
    NonUnit,
    /// Unit diagonal (diagonal elements are treated as 1.0)
    Unit,
}

impl From<Diagonal> for u32 {
    fn from(diag: Diagonal) -> Self {
        match diag {
            Diagonal::NonUnit => bindings::rocblas_diagonal__rocblas_diagonal_non_unit,
            Diagonal::Unit => bindings::rocblas_diagonal__rocblas_diagonal_unit,
        }
    }
}

impl From<u32> for Diagonal {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_diagonal__rocblas_diagonal_non_unit => Diagonal::NonUnit,
            bindings::rocblas_diagonal__rocblas_diagonal_unit => Diagonal::Unit,
            _ => panic!("Invalid rocblas_diagonal value"),
        }
    }
}

/// Side for matrix operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Side {
    /// Apply operation from the left
    Left,
    /// Apply operation from the right
    Right,
    /// Apply operation to both sides
    Both,
}

impl From<Side> for u32 {
    fn from(side: Side) -> Self {
        match side {
            Side::Left => bindings::rocblas_side__rocblas_side_left,
            Side::Right => bindings::rocblas_side__rocblas_side_right,
            Side::Both => bindings::rocblas_side__rocblas_side_both,
        }
    }
}

impl From<u32> for Side {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_side__rocblas_side_left => Side::Left,
            bindings::rocblas_side__rocblas_side_right => Side::Right,
            bindings::rocblas_side__rocblas_side_both => Side::Both,
            _ => panic!("Invalid rocblas_side value"),
        }
    }
}

/// Data type for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

impl From<DataType> for u32 {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::F16Real => bindings::rocblas_datatype__rocblas_datatype_f16_r,
            DataType::F32Real => bindings::rocblas_datatype__rocblas_datatype_f32_r,
            DataType::F64Real => bindings::rocblas_datatype__rocblas_datatype_f64_r,
            DataType::F16Complex => bindings::rocblas_datatype__rocblas_datatype_f16_c,
            DataType::F32Complex => bindings::rocblas_datatype__rocblas_datatype_f32_c,
            DataType::F64Complex => bindings::rocblas_datatype__rocblas_datatype_f64_c,
            DataType::I8Real => bindings::rocblas_datatype__rocblas_datatype_i8_r,
            DataType::U8Real => bindings::rocblas_datatype__rocblas_datatype_u8_r,
            DataType::I32Real => bindings::rocblas_datatype__rocblas_datatype_i32_r,
            DataType::U32Real => bindings::rocblas_datatype__rocblas_datatype_u32_r,
            DataType::I8Complex => bindings::rocblas_datatype__rocblas_datatype_i8_c,
            DataType::U8Complex => bindings::rocblas_datatype__rocblas_datatype_u8_c,
            DataType::I32Complex => bindings::rocblas_datatype__rocblas_datatype_i32_c,
            DataType::U32Complex => bindings::rocblas_datatype__rocblas_datatype_u32_c,
            DataType::BF16Real => bindings::rocblas_datatype__rocblas_datatype_bf16_r,
            DataType::BF16Complex => bindings::rocblas_datatype__rocblas_datatype_bf16_c,
            DataType::F8Real => bindings::rocblas_datatype__rocblas_datatype_f8_r,
            DataType::BF8Real => bindings::rocblas_datatype__rocblas_datatype_bf8_r,
        }
    }
}

impl From<u32> for DataType {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_datatype__rocblas_datatype_f16_r => DataType::F16Real,
            bindings::rocblas_datatype__rocblas_datatype_f32_r => DataType::F32Real,
            bindings::rocblas_datatype__rocblas_datatype_f64_r => DataType::F64Real,
            bindings::rocblas_datatype__rocblas_datatype_f16_c => DataType::F16Complex,
            bindings::rocblas_datatype__rocblas_datatype_f32_c => DataType::F32Complex,
            bindings::rocblas_datatype__rocblas_datatype_f64_c => DataType::F64Complex,
            bindings::rocblas_datatype__rocblas_datatype_i8_r => DataType::I8Real,
            bindings::rocblas_datatype__rocblas_datatype_u8_r => DataType::U8Real,
            bindings::rocblas_datatype__rocblas_datatype_i32_r => DataType::I32Real,
            bindings::rocblas_datatype__rocblas_datatype_u32_r => DataType::U32Real,
            bindings::rocblas_datatype__rocblas_datatype_i8_c => DataType::I8Complex,
            bindings::rocblas_datatype__rocblas_datatype_u8_c => DataType::U8Complex,
            bindings::rocblas_datatype__rocblas_datatype_i32_c => DataType::I32Complex,
            bindings::rocblas_datatype__rocblas_datatype_u32_c => DataType::U32Complex,
            bindings::rocblas_datatype__rocblas_datatype_bf16_r => DataType::BF16Real,
            bindings::rocblas_datatype__rocblas_datatype_bf16_c => DataType::BF16Complex,
            bindings::rocblas_datatype__rocblas_datatype_f8_r => DataType::F8Real,
            bindings::rocblas_datatype__rocblas_datatype_bf8_r => DataType::BF8Real,
            _ => panic!("Invalid rocblas_datatype value"),
        }
    }
}

/// Compute type for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ComputeType {
    /// F32 computation
    F32,
    /// F8-F8-F32 computation
    F8F8F32,
    /// F8-BF8-F32 computation
    F8BF8F32,
    /// BF8-F8-F32 computation
    BF8F8F32,
    /// BF8-BF8-F32 computation
    BF8BF8F32,
}

impl From<ComputeType> for u32 {
    fn from(ct: ComputeType) -> Self {
        match ct {
            ComputeType::F32 => bindings::rocblas_computetype__rocblas_compute_type_f32,
            ComputeType::F8F8F32 => bindings::rocblas_computetype__rocblas_compute_type_f8_f8_f32,
            ComputeType::F8BF8F32 => bindings::rocblas_computetype__rocblas_compute_type_f8_bf8_f32,
            ComputeType::BF8F8F32 => bindings::rocblas_computetype__rocblas_compute_type_bf8_f8_f32,
            ComputeType::BF8BF8F32 => {
                bindings::rocblas_computetype__rocblas_compute_type_bf8_bf8_f32
            }
        }
    }
}

impl From<u32> for ComputeType {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_computetype__rocblas_compute_type_f32 => ComputeType::F32,
            bindings::rocblas_computetype__rocblas_compute_type_f8_f8_f32 => ComputeType::F8F8F32,
            bindings::rocblas_computetype__rocblas_compute_type_f8_bf8_f32 => ComputeType::F8BF8F32,
            bindings::rocblas_computetype__rocblas_compute_type_bf8_f8_f32 => ComputeType::BF8F8F32,
            bindings::rocblas_computetype__rocblas_compute_type_bf8_bf8_f32 => {
                ComputeType::BF8BF8F32
            }
            _ => panic!("Invalid rocblas_computetype value"),
        }
    }
}

/// Pointer mode for rocBLAS scalar parameters
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PointerMode {
    /// Scalar values are on the host
    Host,
    /// Scalar values are on the device
    Device,
}

impl From<PointerMode> for u32 {
    fn from(pm: PointerMode) -> Self {
        match pm {
            PointerMode::Host => bindings::rocblas_pointer_mode__rocblas_pointer_mode_host,
            PointerMode::Device => bindings::rocblas_pointer_mode__rocblas_pointer_mode_device,
        }
    }
}

impl From<u32> for PointerMode {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_pointer_mode__rocblas_pointer_mode_host => PointerMode::Host,
            bindings::rocblas_pointer_mode__rocblas_pointer_mode_device => PointerMode::Device,
            _ => panic!("Invalid rocblas_pointer_mode value"),
        }
    }
}

/// Atomics mode for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AtomicsMode {
    /// Atomics are not allowed
    NotAllowed,
    /// Atomics are allowed
    Allowed,
}

impl From<AtomicsMode> for u32 {
    fn from(am: AtomicsMode) -> Self {
        match am {
            AtomicsMode::NotAllowed => bindings::rocblas_atomics_mode__rocblas_atomics_not_allowed,
            AtomicsMode::Allowed => bindings::rocblas_atomics_mode__rocblas_atomics_allowed,
        }
    }
}

impl From<u32> for AtomicsMode {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_atomics_mode__rocblas_atomics_not_allowed => AtomicsMode::NotAllowed,
            bindings::rocblas_atomics_mode__rocblas_atomics_allowed => AtomicsMode::Allowed,
            _ => panic!("Invalid rocblas_atomics_mode value"),
        }
    }
}

/// Performance metrics for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PerformanceMetric {
    /// Use Tensile's default performance metric
    Default,
    /// Select solution with highest GFlops across all compute units
    DeviceEfficiency,
    /// Select solution with highest GFlops per compute unit
    CUEfficiency,
}

impl From<PerformanceMetric> for u32 {
    fn from(pm: PerformanceMetric) -> Self {
        match pm {
            PerformanceMetric::Default => {
                bindings::rocblas_performance_metric__rocblas_default_performance_metric
            }
            PerformanceMetric::DeviceEfficiency => {
                bindings::rocblas_performance_metric__rocblas_device_efficiency_performance_metric
            }
            PerformanceMetric::CUEfficiency => {
                bindings::rocblas_performance_metric__rocblas_cu_efficiency_performance_metric
            }
        }
    }
}

impl From<u32> for PerformanceMetric {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_performance_metric__rocblas_default_performance_metric => {
                PerformanceMetric::Default
            }
            bindings::rocblas_performance_metric__rocblas_device_efficiency_performance_metric => {
                PerformanceMetric::DeviceEfficiency
            }
            bindings::rocblas_performance_metric__rocblas_cu_efficiency_performance_metric => {
                PerformanceMetric::CUEfficiency
            }
            _ => panic!("Invalid rocblas_performance_metric value"),
        }
    }
}

/// Layer mode for rocBLAS logging
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LayerMode(pub u32);

impl LayerMode {
    /// No logging
    pub const NONE: Self = Self(bindings::rocblas_layer_mode__rocblas_layer_mode_none);
    /// Log function name and arguments
    pub const LOG_TRACE: Self = Self(bindings::rocblas_layer_mode__rocblas_layer_mode_log_trace);
    /// Output benchmarking information
    pub const LOG_BENCH: Self = Self(bindings::rocblas_layer_mode__rocblas_layer_mode_log_bench);
    /// Output profile information
    pub const LOG_PROFILE: Self =
        Self(bindings::rocblas_layer_mode__rocblas_layer_mode_log_profile);

    /// Check if the specified mode is enabled
    pub fn has_mode(&self, mode: Self) -> bool {
        (self.0 & mode.0) != 0
    }

    /// Add a mode to the current mode
    pub fn with_mode(&self, mode: Self) -> Self {
        Self(self.0 | mode.0)
    }

    /// Remove a mode from the current mode
    pub fn without_mode(&self, mode: Self) -> Self {
        Self(self.0 & !mode.0)
    }
}

impl From<u32> for LayerMode {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<LayerMode> for u32 {
    fn from(mode: LayerMode) -> Self {
        mode.0
    }
}

/// GEMM algorithm selection
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GemmAlgo {
    /// Standard algorithm selection
    Standard,
    /// Solution index algorithm selection
    SolutionIndex,
}

impl From<GemmAlgo> for u32 {
    fn from(algo: GemmAlgo) -> Self {
        match algo {
            GemmAlgo::Standard => bindings::rocblas_gemm_algo__rocblas_gemm_algo_standard,
            GemmAlgo::SolutionIndex => {
                bindings::rocblas_gemm_algo__rocblas_gemm_algo_solution_index
            }
        }
    }
}

impl From<u32> for GemmAlgo {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_gemm_algo__rocblas_gemm_algo_standard => GemmAlgo::Standard,
            bindings::rocblas_gemm_algo__rocblas_gemm_algo_solution_index => {
                GemmAlgo::SolutionIndex
            }
            _ => panic!("Invalid rocblas_gemm_algo value"),
        }
    }
}

/// GEAM-Ex operation selection
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GeamExOperation {
    /// Min-Plus operation
    MinPlus,
    /// Plus-Min operation
    PlusMin,
}

impl From<GeamExOperation> for u32 {
    fn from(op: GeamExOperation) -> Self {
        match op {
            GeamExOperation::MinPlus => {
                bindings::rocblas_geam_ex_operation__rocblas_geam_ex_operation_min_plus
            }
            GeamExOperation::PlusMin => {
                bindings::rocblas_geam_ex_operation__rocblas_geam_ex_operation_plus_min
            }
        }
    }
}

impl From<u32> for GeamExOperation {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_geam_ex_operation__rocblas_geam_ex_operation_min_plus => {
                GeamExOperation::MinPlus
            }
            bindings::rocblas_geam_ex_operation__rocblas_geam_ex_operation_plus_min => {
                GeamExOperation::PlusMin
            }
            _ => panic!("Invalid rocblas_geam_ex_operation value"),
        }
    }
}

/// Flags for GEMM operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GemmFlags(pub u32);

impl GemmFlags {
    /// Default empty flags
    pub const NONE: Self = Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_none);
    /// Use compute unit efficiency
    pub const USE_CU_EFFICIENCY: Self =
        Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_use_cu_efficiency);
    /// Use FP16 alternative implementation
    pub const FP16_ALT_IMPL: Self =
        Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl);
    /// Check solution index
    pub const CHECK_SOLUTION_INDEX: Self =
        Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_check_solution_index);
    /// Use FP16 alternative implementation with round-to-nearest-zero
    pub const FP16_ALT_IMPL_RNZ: Self =
        Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl_rnz);
    /// Use stochastic rounding
    pub const STOCHASTIC_ROUNDING: Self =
        Self(bindings::rocblas_gemm_flags__rocblas_gemm_flags_stochastic_rounding);

    /// Check if the specified flag is set
    pub fn has_flag(&self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Add a flag to the current flags
    pub fn with_flag(&self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    /// Remove a flag from the current flags
    pub fn without_flag(&self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }
}

impl From<u32> for GemmFlags {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<GemmFlags> for u32 {
    fn from(flags: GemmFlags) -> Self {
        flags.0
    }
}

/// Check numerics mode for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CheckNumericsMode(pub u32);

impl CheckNumericsMode {
    /// No checking for special values
    pub const NO_CHECK: Self =
        Self(bindings::rocblas_check_numerics_mode__rocblas_check_numerics_mode_no_check);
    /// Report information about special values
    pub const INFO: Self =
        Self(bindings::rocblas_check_numerics_mode__rocblas_check_numerics_mode_info);
    /// Warn about special values
    pub const WARN: Self =
        Self(bindings::rocblas_check_numerics_mode__rocblas_check_numerics_mode_warn);
    /// Fail on special values
    pub const FAIL: Self =
        Self(bindings::rocblas_check_numerics_mode__rocblas_check_numerics_mode_fail);
    /// Only check for NaN and infinity
    pub const ONLY_NAN_INF: Self =
        Self(bindings::rocblas_check_numerics_mode__rocblas_check_numerics_mode_only_nan_inf);

    /// Check if the specified mode is enabled
    pub fn has_mode(&self, mode: Self) -> bool {
        (self.0 & mode.0) != 0
    }

    /// Add a mode to the current mode
    pub fn with_mode(&self, mode: Self) -> Self {
        Self(self.0 | mode.0)
    }

    /// Remove a mode from the current mode
    pub fn without_mode(&self, mode: Self) -> Self {
        Self(self.0 & !mode.0)
    }
}

impl From<u32> for CheckNumericsMode {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<CheckNumericsMode> for u32 {
    fn from(mode: CheckNumericsMode) -> Self {
        mode.0
    }
}

/// Math mode for rocBLAS operations
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MathMode {
    /// Default math mode
    DefaultMath,
    /// XF32 XDL math operations
    XF32XDLMathOp,
}

impl From<MathMode> for u32 {
    fn from(mode: MathMode) -> Self {
        match mode {
            MathMode::DefaultMath => bindings::rocblas_math_mode__rocblas_default_math,
            MathMode::XF32XDLMathOp => bindings::rocblas_math_mode__rocblas_xf32_xdl_math_op,
        }
    }
}

impl From<u32> for MathMode {
    fn from(value: u32) -> Self {
        match value {
            bindings::rocblas_math_mode__rocblas_default_math => MathMode::DefaultMath,
            bindings::rocblas_math_mode__rocblas_xf32_xdl_math_op => MathMode::XF32XDLMathOp,
            _ => panic!("Invalid rocblas_math_mode value"),
        }
    }
}
