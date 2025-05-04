// src/rocblas/utils.rs

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::Result;

/// Enum for RocBLAS pointer mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerMode {
    /// Scalar values are on the host
    Host,
    /// Scalar values are on the device
    Device,
}

impl From<PointerMode> for ffi::rocblas_pointer_mode {
    fn from(mode: PointerMode) -> Self {
        match mode {
            PointerMode::Host => ffi::rocblas_pointer_mode__rocblas_pointer_mode_host,
            PointerMode::Device => ffi::rocblas_pointer_mode__rocblas_pointer_mode_device,
        }
    }
}

impl From<ffi::rocblas_pointer_mode> for PointerMode {
    fn from(mode: ffi::rocblas_pointer_mode) -> Self {
        match mode {
            ffi::rocblas_pointer_mode__rocblas_pointer_mode_host => PointerMode::Host,
            ffi::rocblas_pointer_mode__rocblas_pointer_mode_device => PointerMode::Device,
            _ => PointerMode::Host, // Default to Host mode for unknown values
        }
    }
}

/// Enum for RocBLAS atomics mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicsMode {
    /// Algorithms will refrain from atomics where applicable
    NotAllowed,
    /// Algorithms will take advantage of atomics where applicable
    Allowed,
}

impl From<AtomicsMode> for ffi::rocblas_atomics_mode {
    fn from(mode: AtomicsMode) -> Self {
        match mode {
            AtomicsMode::NotAllowed => ffi::rocblas_atomics_mode__rocblas_atomics_not_allowed,
            AtomicsMode::Allowed => ffi::rocblas_atomics_mode__rocblas_atomics_allowed,
        }
    }
}

impl From<ffi::rocblas_atomics_mode> for AtomicsMode {
    fn from(mode: ffi::rocblas_atomics_mode) -> Self {
        match mode {
            ffi::rocblas_atomics_mode__rocblas_atomics_not_allowed => AtomicsMode::NotAllowed,
            ffi::rocblas_atomics_mode__rocblas_atomics_allowed => AtomicsMode::Allowed,
            _ => AtomicsMode::Allowed, // Default to Allowed for unknown values
        }
    }
}

/// Enum for RocBLAS performance metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMetric {
    /// Use Tensile's default performance metric
    Default,
    /// Select solution with highest GFlops across all compute units
    DeviceEfficiency,
    /// Select solution with highest GFlops per compute unit
    CUEfficiency,
}

impl From<PerformanceMetric> for ffi::rocblas_performance_metric {
    fn from(metric: PerformanceMetric) -> Self {
        match metric {
            PerformanceMetric::Default => ffi::rocblas_performance_metric__rocblas_default_performance_metric,
            PerformanceMetric::DeviceEfficiency => ffi::rocblas_performance_metric__rocblas_device_efficiency_performance_metric,
            PerformanceMetric::CUEfficiency => ffi::rocblas_performance_metric__rocblas_cu_efficiency_performance_metric,
        }
    }
}

impl From<ffi::rocblas_performance_metric> for PerformanceMetric {
    fn from(metric: ffi::rocblas_performance_metric) -> Self {
        match metric {
            ffi::rocblas_performance_metric__rocblas_default_performance_metric => PerformanceMetric::Default,
            ffi::rocblas_performance_metric__rocblas_device_efficiency_performance_metric => PerformanceMetric::DeviceEfficiency,
            ffi::rocblas_performance_metric__rocblas_cu_efficiency_performance_metric => PerformanceMetric::CUEfficiency,
            _ => PerformanceMetric::Default, // Default for unknown values
        }
    }
}

/// Enum for RocBLAS layer mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMode {
    /// No logging
    None,
    /// Log a trace of function calls
    LogTrace,
    /// Log a line for benchmarking
    LogBench,
    /// Log a YAML description
    LogProfile,
}

impl From<LayerMode> for ffi::rocblas_layer_mode {
    fn from(mode: LayerMode) -> Self {
        match mode {
            LayerMode::None => ffi::rocblas_layer_mode__rocblas_layer_mode_none,
            LayerMode::LogTrace => ffi::rocblas_layer_mode__rocblas_layer_mode_log_trace,
            LayerMode::LogBench => ffi::rocblas_layer_mode__rocblas_layer_mode_log_bench,
            LayerMode::LogProfile => ffi::rocblas_layer_mode__rocblas_layer_mode_log_profile,
        }
    }
}

/// Enum for RocBLAS GEMM algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmAlgo {
    /// Standard algorithm
    Standard,
    /// Algorithm using solution index
    SolutionIndex,
}

impl From<GemmAlgo> for ffi::rocblas_gemm_algo {
    fn from(algo: GemmAlgo) -> Self {
        match algo {
            GemmAlgo::Standard => ffi::rocblas_gemm_algo__rocblas_gemm_algo_standard,
            GemmAlgo::SolutionIndex => ffi::rocblas_gemm_algo__rocblas_gemm_algo_solution_index,
        }
    }
}

impl From<ffi::rocblas_gemm_algo> for GemmAlgo {
    fn from(algo: ffi::rocblas_gemm_algo) -> Self {
        match algo {
            ffi::rocblas_gemm_algo__rocblas_gemm_algo_standard => GemmAlgo::Standard,
            ffi::rocblas_gemm_algo__rocblas_gemm_algo_solution_index => GemmAlgo::SolutionIndex,
            _ => GemmAlgo::Standard, // Default for unknown values
        }
    }
}

/// Enum for RocBLAS GEMM flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmFlags {
    /// Default empty flags
    None,
    /// Use CU efficiency
    UseCUEfficiency,
    /// FP16 alternate implementation
    FP16AltImpl,
    /// Check solution index
    CheckSolutionIndex,
    /// FP16 alternate implementation with round-to-nearest-zero
    FP16AltImplRNZ,
    /// Stochastic rounding
    StochasticRounding,
}

impl From<GemmFlags> for ffi::rocblas_gemm_flags {
    fn from(flags: GemmFlags) -> Self {
        match flags {
            GemmFlags::None => ffi::rocblas_gemm_flags__rocblas_gemm_flags_none,
            GemmFlags::UseCUEfficiency => ffi::rocblas_gemm_flags__rocblas_gemm_flags_use_cu_efficiency,
            GemmFlags::FP16AltImpl => ffi::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl,
            GemmFlags::CheckSolutionIndex => ffi::rocblas_gemm_flags__rocblas_gemm_flags_check_solution_index,
            GemmFlags::FP16AltImplRNZ => ffi::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl_rnz,
            GemmFlags::StochasticRounding => ffi::rocblas_gemm_flags__rocblas_gemm_flags_stochastic_rounding,
        }
    }
}

/// Enum for RocBLAS math mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathMode {
    /// Default math mode
    Default,
    /// XF32 XDL math operation
    XF32XDLMathOp,
}

impl From<MathMode> for ffi::rocblas_math_mode {
    fn from(mode: MathMode) -> Self {
        match mode {
            MathMode::Default => ffi::rocblas_math_mode__rocblas_default_math,
            MathMode::XF32XDLMathOp => ffi::rocblas_math_mode__rocblas_xf32_xdl_math_op,
        }
    }
}

impl From<ffi::rocblas_math_mode> for MathMode {
    fn from(mode: ffi::rocblas_math_mode) -> Self {
        match mode {
            ffi::rocblas_math_mode__rocblas_default_math => MathMode::Default,
            ffi::rocblas_math_mode__rocblas_xf32_xdl_math_op => MathMode::XF32XDLMathOp,
            _ => MathMode::Default, // Default for unknown values
        }
    }
}

/// Set the pointer mode for a RocBLAS handle
pub fn set_pointer_mode(handle: &Handle, mode: PointerMode) -> Result<()> {
    handle.set_pointer_mode(mode.into())
}

/// Get the pointer mode for a RocBLAS handle
pub fn get_pointer_mode(handle: &Handle) -> Result<PointerMode> {
    let mode = handle.get_pointer_mode()?;
    Ok(mode.into())
}

/// Set the atomics mode for a RocBLAS handle
pub fn set_atomics_mode(handle: &Handle, mode: AtomicsMode) -> Result<()> {
    handle.set_atomics_mode(mode.into())
}

/// Get the atomics mode for a RocBLAS handle
pub fn get_atomics_mode(handle: &Handle) -> Result<AtomicsMode> {
    let mode = handle.get_atomics_mode()?;
    Ok(mode.into())
}

/// Set the performance metric for a RocBLAS handle
pub fn set_performance_metric(handle: &Handle, metric: PerformanceMetric) -> Result<()> {
    handle.set_performance_metric(metric.into())
}

/// Get the performance metric for a RocBLAS handle
pub fn get_performance_metric(handle: &Handle) -> Result<PerformanceMetric> {
    let metric = handle.get_performance_metric()?;
    Ok(metric.into())
}

/// Set the math mode for a RocBLAS handle
pub fn set_math_mode(handle: &Handle, mode: MathMode) -> Result<()> {
    handle.set_math_mode(mode.into())
}

/// Get the math mode for a RocBLAS handle
pub fn get_math_mode(handle: &Handle) -> Result<MathMode> {
    let mode = handle.get_math_mode()?;
    Ok(mode.into())
}