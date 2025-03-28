// FFI module for rocBLAS
// This file re-exports the necessary symbols from the auto-generated bindings

// Import the raw bindings from the auto-generated module
use crate::rocblas::bindings;

// Re-export the necessary types, constants, and functions

// Handle type
pub use bindings::rocblas_handle;

// Basic types
pub use bindings::{
    rocblas_int,
    rocblas_stride,
    rocblas_float,
    rocblas_double,
    rocblas_half,
    rocblas_float_complex,
    rocblas_double_complex,
    rocblas_bfloat16,
    rocblas_f8,
    rocblas_bf8,
};

// Status type
pub use bindings::rocblas_status_ as rocblas_status_t;
pub use bindings::rocblas_status as rocblas_status_t_alias;

// Enum types
pub use bindings::{
    rocblas_operation,
    rocblas_fill,
    rocblas_diagonal,
    rocblas_side,
    rocblas_datatype,
    rocblas_computetype,
    rocblas_pointer_mode,
    rocblas_atomics_mode,
    rocblas_performance_metric,
    rocblas_layer_mode,
    rocblas_gemm_algo,
    rocblas_gemm_flags,
    rocblas_check_numerics_mode,
    rocblas_math_mode,
    rocblas_geam_ex_operation,
    rocblas_destroy_handle,
    rocblas_set_vector,
    rocblas_get_vector,
    rocblas_get_vector_64,
    rocblas_set_matrix,
    rocblas_set_matrix_64,
    rocblas_set_vector_64,
    rocblas_get_matrix_async,
    rocblas_set_performance_metric,
    rocblas_get_performance_metric,
    rocblas_get_math_mode,
    rocblas_set_math_mode,
    rocblas_get_atomics_mode,
    rocblas_set_atomics_mode,
    rocblas_get_vector_async,
    rocblas_get_pointer_mode,
    rocblas_set_pointer_mode,
    rocblas_get_stream,
    rocblas_set_stream,
    rocblas_set_vector_async,
    rocblas_create_handle,
    rocblas_get_matrix,
    rocblas_get_matrix_64,
    rocblas_set_matrix_async
};

// Status constants
pub use bindings::{
    rocblas_status__rocblas_status_success as STATUS_SUCCESS,
    rocblas_status__rocblas_status_invalid_handle as STATUS_INVALID_HANDLE,
    rocblas_status__rocblas_status_not_implemented as STATUS_NOT_IMPLEMENTED,
    rocblas_status__rocblas_status_invalid_pointer as STATUS_INVALID_POINTER,
    rocblas_status__rocblas_status_invalid_size as STATUS_INVALID_SIZE,
    rocblas_status__rocblas_status_memory_error as STATUS_MEMORY_ERROR,
    rocblas_status__rocblas_status_internal_error as STATUS_INTERNAL_ERROR,
    rocblas_status__rocblas_status_perf_degraded as STATUS_PERF_DEGRADED,
    rocblas_status__rocblas_status_size_query_mismatch as STATUS_SIZE_QUERY_MISMATCH,
    rocblas_status__rocblas_status_size_increased as STATUS_SIZE_INCREASED,
    rocblas_status__rocblas_status_size_unchanged as STATUS_SIZE_UNCHANGED,
    rocblas_status__rocblas_status_invalid_value as STATUS_INVALID_VALUE,
    rocblas_status__rocblas_status_continue as STATUS_CONTINUE,
    rocblas_status__rocblas_status_check_numerics_fail as STATUS_CHECK_NUMERICS_FAIL,
    rocblas_status__rocblas_status_excluded_from_build as STATUS_EXCLUDED_FROM_BUILD,
    rocblas_status__rocblas_status_arch_mismatch as STATUS_ARCH_MISMATCH,
};

// Operation constants
pub use bindings::{
    rocblas_operation__rocblas_operation_none as OPERATION_NONE,
    rocblas_operation__rocblas_operation_transpose as OPERATION_TRANSPOSE,
    rocblas_operation__rocblas_operation_conjugate_transpose as OPERATION_CONJUGATE_TRANSPOSE,
};

// Fill constants
pub use bindings::{
    rocblas_fill__rocblas_fill_upper as FILL_UPPER,
    rocblas_fill__rocblas_fill_lower as FILL_LOWER,
    rocblas_fill__rocblas_fill_full as FILL_FULL,
};

// Diagonal constants
pub use bindings::{
    rocblas_diagonal__rocblas_diagonal_non_unit as DIAGONAL_NON_UNIT,
    rocblas_diagonal__rocblas_diagonal_unit as DIAGONAL_UNIT,
};

// Side constants
pub use bindings::{
    rocblas_side__rocblas_side_left as SIDE_LEFT,
    rocblas_side__rocblas_side_right as SIDE_RIGHT,
    rocblas_side__rocblas_side_both as SIDE_BOTH,
};

// Pointer mode constants
pub use bindings::{
    rocblas_pointer_mode__rocblas_pointer_mode_host as POINTER_MODE_HOST,
    rocblas_pointer_mode__rocblas_pointer_mode_device as POINTER_MODE_DEVICE,
};

// Atomics mode constants
pub use bindings::{
    rocblas_atomics_mode__rocblas_atomics_not_allowed as ATOMICS_NOT_ALLOWED,
    rocblas_atomics_mode__rocblas_atomics_allowed as ATOMICS_ALLOWED,
};

// Performance metric constants
pub use bindings::{
    rocblas_performance_metric__rocblas_default_performance_metric as PERFORMANCE_METRIC_DEFAULT,
    rocblas_performance_metric__rocblas_device_efficiency_performance_metric as PERFORMANCE_METRIC_DEVICE_EFFICIENCY,
    rocblas_performance_metric__rocblas_cu_efficiency_performance_metric as PERFORMANCE_METRIC_CU_EFFICIENCY,
};

// Layer mode constants
pub use bindings::{
    rocblas_layer_mode__rocblas_layer_mode_none as LAYER_MODE_NONE,
    rocblas_layer_mode__rocblas_layer_mode_log_trace as LAYER_MODE_LOG_TRACE,
    rocblas_layer_mode__rocblas_layer_mode_log_bench as LAYER_MODE_LOG_BENCH,
    rocblas_layer_mode__rocblas_layer_mode_log_profile as LAYER_MODE_LOG_PROFILE,
};

// GEMM algorithm constants
pub use bindings::{
    rocblas_gemm_algo__rocblas_gemm_algo_standard as GEMM_ALGO_STANDARD,
    rocblas_gemm_algo__rocblas_gemm_algo_solution_index as GEMM_ALGO_SOLUTION_INDEX,
};

// GEAM Ex operation constants
pub use bindings::{
    rocblas_geam_ex_operation__rocblas_geam_ex_operation_min_plus as GEAM_EX_OPERATION_MIN_PLUS,
    rocblas_geam_ex_operation__rocblas_geam_ex_operation_plus_min as GEAM_EX_OPERATION_PLUS_MIN,
};

// GEMM flags constants
pub use bindings::{
    rocblas_gemm_flags__rocblas_gemm_flags_none as GEMM_FLAGS_NONE,
    rocblas_gemm_flags__rocblas_gemm_flags_use_cu_efficiency as GEMM_FLAGS_USE_CU_EFFICIENCY,
    rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl as GEMM_FLAGS_FP16_ALT_IMPL,
    rocblas_gemm_flags__rocblas_gemm_flags_check_solution_index as GEMM_FLAGS_CHECK_SOLUTION_INDEX,
    rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl_rnz as GEMM_FLAGS_FP16_ALT_IMPL_RNZ,
    rocblas_gemm_flags__rocblas_gemm_flags_stochastic_rounding as GEMM_FLAGS_STOCHASTIC_ROUNDING,
};

// Check numerics mode constants
pub use bindings::{
    rocblas_check_numerics_mode__rocblas_check_numerics_mode_no_check as CHECK_NUMERICS_MODE_NO_CHECK,
    rocblas_check_numerics_mode__rocblas_check_numerics_mode_info as CHECK_NUMERICS_MODE_INFO,
    rocblas_check_numerics_mode__rocblas_check_numerics_mode_warn as CHECK_NUMERICS_MODE_WARN,
    rocblas_check_numerics_mode__rocblas_check_numerics_mode_fail as CHECK_NUMERICS_MODE_FAIL,
    rocblas_check_numerics_mode__rocblas_check_numerics_mode_only_nan_inf as CHECK_NUMERICS_MODE_ONLY_NAN_INF,
};

// Math mode constants
pub use bindings::{
    rocblas_math_mode__rocblas_default_math as MATH_MODE_DEFAULT,
    rocblas_math_mode__rocblas_xf32_xdl_math_op as MATH_MODE_XF32_XDL,
};

// Datatype constants
pub use bindings::{
    rocblas_datatype__rocblas_datatype_f16_r as DATATYPE_F16_R,
    rocblas_datatype__rocblas_datatype_f32_r as DATATYPE_F32_R,
    rocblas_datatype__rocblas_datatype_f64_r as DATATYPE_F64_R,
    rocblas_datatype__rocblas_datatype_f16_c as DATATYPE_F16_C,
    rocblas_datatype__rocblas_datatype_f32_c as DATATYPE_F32_C,
    rocblas_datatype__rocblas_datatype_f64_c as DATATYPE_F64_C,
    rocblas_datatype__rocblas_datatype_i8_r as DATATYPE_I8_R,
    rocblas_datatype__rocblas_datatype_u8_r as DATATYPE_U8_R,
    rocblas_datatype__rocblas_datatype_i32_r as DATATYPE_I32_R,
    rocblas_datatype__rocblas_datatype_u32_r as DATATYPE_U32_R,
    rocblas_datatype__rocblas_datatype_i8_c as DATATYPE_I8_C,
    rocblas_datatype__rocblas_datatype_u8_c as DATATYPE_U8_C,
    rocblas_datatype__rocblas_datatype_i32_c as DATATYPE_I32_C,
    rocblas_datatype__rocblas_datatype_u32_c as DATATYPE_U32_C,
    rocblas_datatype__rocblas_datatype_bf16_r as DATATYPE_BF16_R,
    rocblas_datatype__rocblas_datatype_bf16_c as DATATYPE_BF16_C,
    rocblas_datatype__rocblas_datatype_f8_r as DATATYPE_F8_R,
    rocblas_datatype__rocblas_datatype_bf8_r as DATATYPE_BF8_R,
    rocblas_datatype__rocblas_datatype_invalid as DATATYPE_INVALID,
};

// Compute type constants
pub use bindings::{
    rocblas_computetype__rocblas_compute_type_f32 as COMPUTE_TYPE_F32,
    rocblas_computetype__rocblas_compute_type_f8_f8_f32 as COMPUTE_TYPE_F8_F8_F32,
    rocblas_computetype__rocblas_compute_type_f8_bf8_f32 as COMPUTE_TYPE_F8_BF8_F32,
    rocblas_computetype__rocblas_compute_type_bf8_f8_f32 as COMPUTE_TYPE_BF8_F8_F32,
    rocblas_computetype__rocblas_compute_type_bf8_bf8_f32 as COMPUTE_TYPE_BF8_BF8_F32,
    rocblas_computetype__rocblas_compute_type_invalid as COMPUTE_TYPE_INVALID,
};