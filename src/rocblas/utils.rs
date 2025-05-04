// src/rocblas/utils.rs

use std::ffi::CStr;

use crate::rocblas::ffi;
use crate::rocblas::handle::Handle;
use crate::rocblas::error::Result;

use super::Error;

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

// src/rocblas/utils.rs or appropriate file


/// Convert a rocBLAS status code to a string representation
pub fn status_to_string(status: ffi::rocblas_status) -> String {
    unsafe {
        let c_str = ffi::rocblas_status_to_string(status);
        if c_str.is_null() {
            return String::from("Unknown rocBLAS status");
        }
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

/// Initialize rocBLAS on the current HIP device
///
/// This function can be called to initialize rocBLAS upfront, 
/// avoiding costly startup time at the first function call.
/// Otherwise, initialization happens automatically on the first call.
pub fn initialize() {
    unsafe {
        ffi::rocblas_initialize();
    }
}

/// Get the rocBLAS library version as a string
pub fn get_version_string() -> Result<String> {
    // First, get the required buffer size
    let mut size: usize = 0;
    let status = unsafe {
        ffi::rocblas_get_version_string_size(&mut size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    // Allocate buffer and get the version string
    let mut buffer = vec![0u8; size];
    let status = unsafe {
        ffi::rocblas_get_version_string(buffer.as_mut_ptr() as *mut i8, size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }

    // Convert to Rust string
    Ok(String::from_utf8_lossy(&buffer[..buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len())]).into_owned())
}

/// Start collecting optimal device memory size information
///
/// Indicates that subsequent rocBLAS kernel calls should collect 
/// the optimal device memory size in bytes for their given kernel arguments
/// and keep track of the maximum.
pub fn start_device_memory_size_query(handle: &Handle) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_start_device_memory_size_query(handle.as_raw())
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Stop collecting optimal device memory size information
pub fn stop_device_memory_size_query(handle: &Handle) -> Result<usize> {
    let mut size: usize = 0;
    let status = unsafe {
        ffi::rocblas_stop_device_memory_size_query(handle.as_raw(), &mut size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(size)
}

/// Check if device memory size query is in progress
pub fn is_device_memory_size_query(handle: &Handle) -> bool {
    unsafe {
        ffi::rocblas_is_device_memory_size_query(handle.as_raw())
    }
}

/// Get the current device memory size for the handle
pub fn get_device_memory_size(handle: &Handle) -> Result<usize> {
    let mut size: usize = 0;
    let status = unsafe {
        ffi::rocblas_get_device_memory_size(handle.as_raw(), &mut size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(size)
}

/// Set the size of allocated device memory
///
/// Changes the size of allocated device memory at runtime.
///
/// Any previously allocated device memory managed by the handle is freed.
///
/// If size > 0, sets the device memory size to the specified size (in bytes).
/// If size == 0, frees the memory allocated so far, and lets rocBLAS manage 
/// device memory in the future, expanding it when necessary.
pub fn set_device_memory_size(handle: &Handle, size: usize) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_set_device_memory_size(handle.as_raw(), size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Set the device workspace for the handle to use
///
/// Any previously allocated device memory managed by the handle is freed.
pub fn set_workspace(handle: &Handle, addr: *mut std::ffi::c_void, size: usize) -> Result<()> {
    let status = unsafe {
        ffi::rocblas_set_workspace(handle.as_raw(), addr, size)
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(Error::new(status));
    }
    Ok(())
}

/// Check if device memory in handle is managed by rocBLAS
pub fn is_managing_device_memory(handle: &Handle) -> bool {
    unsafe {
        ffi::rocblas_is_managing_device_memory(handle.as_raw())
    }
}

/// Check if device memory in handle is managed by the user
pub fn is_user_managing_device_memory(handle: &Handle) -> bool {
    unsafe {
        ffi::rocblas_is_user_managing_device_memory(handle.as_raw())
    }
}

/// Set the default memory size for device malloc
pub fn device_malloc_set_default_memory_size(size: usize) {
    unsafe {
        ffi::rocblas_device_malloc_set_default_memory_size(size);
    }
}

/// Abort rocBLAS execution immediately
pub fn abort() -> ! {
    unsafe {
        ffi::rocblas_abort();
    }
}