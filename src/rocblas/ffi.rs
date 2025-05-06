// src/rocblas/ffi.rs
//
// FFI bindings for the RocBLAS API
// This file re-exports the necessary symbols from the auto-generated bindings

// We assume there's a bindings module that was auto-generated
// using bindgen or similar tool
use crate::rocblas::bindings;

// Re-export the necessary types, constants, and functions

// Basic types
pub use bindings::rocblas_handle;
pub use bindings::rocblas_half;
pub use bindings::rocblas_bfloat16;
pub use bindings::rocblas_int;
pub use bindings::rocblas_stride;
pub use bindings::rocblas_float;
pub use bindings::rocblas_double;
pub use bindings::rocblas_float_complex;
pub use bindings::rocblas_double_complex;

// Status type and constants
pub use bindings::rocblas_status;
pub use bindings::rocblas_status__rocblas_status_success;
pub use bindings::rocblas_status__rocblas_status_invalid_handle;
pub use bindings::rocblas_status__rocblas_status_not_implemented;
pub use bindings::rocblas_status__rocblas_status_invalid_pointer;
pub use bindings::rocblas_status__rocblas_status_invalid_size;
pub use bindings::rocblas_status__rocblas_status_memory_error;
pub use bindings::rocblas_status__rocblas_status_internal_error;
pub use bindings::rocblas_status__rocblas_status_perf_degraded;
pub use bindings::rocblas_status__rocblas_status_size_query_mismatch;
pub use bindings::rocblas_status__rocblas_status_size_increased;
pub use bindings::rocblas_status__rocblas_status_size_unchanged;
pub use bindings::rocblas_status__rocblas_status_invalid_value;
pub use bindings::rocblas_status__rocblas_status_continue;
pub use bindings::rocblas_status__rocblas_status_check_numerics_fail;
pub use bindings::rocblas_status__rocblas_status_excluded_from_build;
pub use bindings::rocblas_status__rocblas_status_arch_mismatch;

// Operation type and constants
pub use bindings::rocblas_operation;
pub use bindings::rocblas_operation__rocblas_operation_none;
pub use bindings::rocblas_operation__rocblas_operation_transpose;
pub use bindings::rocblas_operation__rocblas_operation_conjugate_transpose;

// Fill mode type and constants
pub use bindings::rocblas_fill;
pub use bindings::rocblas_fill__rocblas_fill_upper;
pub use bindings::rocblas_fill__rocblas_fill_lower;
pub use bindings::rocblas_fill__rocblas_fill_full;

// Diagonal type and constants
pub use bindings::rocblas_diagonal;
pub use bindings::rocblas_diagonal__rocblas_diagonal_non_unit;
pub use bindings::rocblas_diagonal__rocblas_diagonal_unit;

// Side type and constants
pub use bindings::rocblas_side;
pub use bindings::rocblas_side__rocblas_side_left;
pub use bindings::rocblas_side__rocblas_side_right;
pub use bindings::rocblas_side__rocblas_side_both;

// Data type constants
pub use bindings::rocblas_datatype;
pub use bindings::rocblas_datatype__rocblas_datatype_f16_r;
pub use bindings::rocblas_datatype__rocblas_datatype_f32_r;
pub use bindings::rocblas_datatype__rocblas_datatype_f64_r;
pub use bindings::rocblas_datatype__rocblas_datatype_f16_c;
pub use bindings::rocblas_datatype__rocblas_datatype_f32_c;
pub use bindings::rocblas_datatype__rocblas_datatype_f64_c;
pub use bindings::rocblas_datatype__rocblas_datatype_i8_r;
pub use bindings::rocblas_datatype__rocblas_datatype_u8_r;
pub use bindings::rocblas_datatype__rocblas_datatype_i32_r;
pub use bindings::rocblas_datatype__rocblas_datatype_u32_r;
pub use bindings::rocblas_datatype__rocblas_datatype_i8_c;
pub use bindings::rocblas_datatype__rocblas_datatype_u8_c;
pub use bindings::rocblas_datatype__rocblas_datatype_i32_c;
pub use bindings::rocblas_datatype__rocblas_datatype_u32_c;
pub use bindings::rocblas_datatype__rocblas_datatype_bf16_r;
pub use bindings::rocblas_datatype__rocblas_datatype_bf16_c;
pub use bindings::rocblas_datatype__rocblas_datatype_f8_r;
pub use bindings::rocblas_datatype__rocblas_datatype_bf8_r;
pub use bindings::rocblas_datatype__rocblas_datatype_invalid;

// Compute type constants
pub use bindings::rocblas_computetype;
pub use bindings::rocblas_computetype__rocblas_compute_type_f32;
pub use bindings::rocblas_computetype__rocblas_compute_type_f8_f8_f32;
pub use bindings::rocblas_computetype__rocblas_compute_type_f8_bf8_f32;
pub use bindings::rocblas_computetype__rocblas_compute_type_bf8_f8_f32;
pub use bindings::rocblas_computetype__rocblas_compute_type_bf8_bf8_f32;
pub use bindings::rocblas_computetype__rocblas_compute_type_invalid;

// Pointer mode type and constants
pub use bindings::rocblas_pointer_mode;
pub use bindings::rocblas_pointer_mode__rocblas_pointer_mode_host;
pub use bindings::rocblas_pointer_mode__rocblas_pointer_mode_device;

// Atomics mode type and constants
pub use bindings::rocblas_atomics_mode;
pub use bindings::rocblas_atomics_mode__rocblas_atomics_not_allowed;
pub use bindings::rocblas_atomics_mode__rocblas_atomics_allowed;

// Performance metric type and constants
pub use bindings::rocblas_performance_metric;
pub use bindings::rocblas_performance_metric__rocblas_default_performance_metric;
pub use bindings::rocblas_performance_metric__rocblas_device_efficiency_performance_metric;
pub use bindings::rocblas_performance_metric__rocblas_cu_efficiency_performance_metric;

// Layer mode type and constants
pub use bindings::rocblas_layer_mode;
pub use bindings::rocblas_layer_mode__rocblas_layer_mode_none;
pub use bindings::rocblas_layer_mode__rocblas_layer_mode_log_trace;
pub use bindings::rocblas_layer_mode__rocblas_layer_mode_log_bench;
pub use bindings::rocblas_layer_mode__rocblas_layer_mode_log_profile;

// GEMM algorithm type and constants
pub use bindings::rocblas_gemm_algo;
pub use bindings::rocblas_gemm_algo__rocblas_gemm_algo_standard;
pub use bindings::rocblas_gemm_algo__rocblas_gemm_algo_solution_index;

// GEMM flags type and constants
pub use bindings::rocblas_gemm_flags;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_none;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_use_cu_efficiency;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_check_solution_index;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_fp16_alt_impl_rnz;
pub use bindings::rocblas_gemm_flags__rocblas_gemm_flags_stochastic_rounding;

// Math mode type and constants
pub use bindings::rocblas_math_mode;
pub use bindings::rocblas_math_mode__rocblas_default_math;
pub use bindings::rocblas_math_mode__rocblas_xf32_xdl_math_op;

// Handle management
pub use bindings::rocblas_create_handle;
pub use bindings::rocblas_destroy_handle;
pub use bindings::rocblas_set_stream;
pub use bindings::rocblas_get_stream;

// Configuration
pub use bindings::rocblas_set_pointer_mode;
pub use bindings::rocblas_get_pointer_mode;
pub use bindings::rocblas_set_atomics_mode;
pub use bindings::rocblas_get_atomics_mode;
pub use bindings::rocblas_set_math_mode;
pub use bindings::rocblas_get_math_mode;
pub use bindings::rocblas_set_performance_metric;
pub use bindings::rocblas_get_performance_metric;

// Level 1 BLAS
pub use bindings::rocblas_sscal;
pub use bindings::rocblas_dscal;
pub use bindings::rocblas_cscal;
pub use bindings::rocblas_zscal;
pub use bindings::rocblas_csscal;
pub use bindings::rocblas_zdscal;

pub use bindings::rocblas_sscal_batched;
pub use bindings::rocblas_dscal_batched;
pub use bindings::rocblas_cscal_batched;
pub use bindings::rocblas_zscal_batched;
pub use bindings::rocblas_csscal_batched;
pub use bindings::rocblas_zdscal_batched;

pub use bindings::rocblas_sscal_strided_batched;
pub use bindings::rocblas_dscal_strided_batched;
pub use bindings::rocblas_cscal_strided_batched;
pub use bindings::rocblas_zscal_strided_batched;
pub use bindings::rocblas_csscal_strided_batched;
pub use bindings::rocblas_zdscal_strided_batched;

pub use bindings::rocblas_scopy;
pub use bindings::rocblas_dcopy;
pub use bindings::rocblas_ccopy;
pub use bindings::rocblas_zcopy;

pub use bindings::rocblas_scopy_batched;
pub use bindings::rocblas_dcopy_batched;
pub use bindings::rocblas_ccopy_batched;
pub use bindings::rocblas_zcopy_batched;

pub use bindings::rocblas_scopy_strided_batched;
pub use bindings::rocblas_dcopy_strided_batched;
pub use bindings::rocblas_ccopy_strided_batched;
pub use bindings::rocblas_zcopy_strided_batched;

pub use bindings::rocblas_sdot;
pub use bindings::rocblas_ddot;
pub use bindings::rocblas_hdot;
pub use bindings::rocblas_bfdot;
pub use bindings::rocblas_cdotu;
pub use bindings::rocblas_zdotu;
pub use bindings::rocblas_cdotc;
pub use bindings::rocblas_zdotc;

pub use bindings::rocblas_sdot_batched;
pub use bindings::rocblas_ddot_batched;
pub use bindings::rocblas_hdot_batched;
pub use bindings::rocblas_bfdot_batched;
pub use bindings::rocblas_cdotu_batched;
pub use bindings::rocblas_zdotu_batched;
pub use bindings::rocblas_cdotc_batched;
pub use bindings::rocblas_zdotc_batched;

pub use bindings::rocblas_sdot_strided_batched;
pub use bindings::rocblas_ddot_strided_batched;
pub use bindings::rocblas_hdot_strided_batched;
pub use bindings::rocblas_bfdot_strided_batched;
pub use bindings::rocblas_cdotu_strided_batched;
pub use bindings::rocblas_zdotu_strided_batched;
pub use bindings::rocblas_cdotc_strided_batched;
pub use bindings::rocblas_zdotc_strided_batched;

pub use bindings::rocblas_saxpy;
pub use bindings::rocblas_daxpy;
pub use bindings::rocblas_caxpy;
pub use bindings::rocblas_zaxpy;

pub use bindings::rocblas_saxpy_batched;
pub use bindings::rocblas_daxpy_batched;
pub use bindings::rocblas_caxpy_batched;
pub use bindings::rocblas_zaxpy_batched;

pub use bindings::rocblas_saxpy_strided_batched;
pub use bindings::rocblas_daxpy_strided_batched;
pub use bindings::rocblas_caxpy_strided_batched;
pub use bindings::rocblas_zaxpy_strided_batched;

pub use bindings::rocblas_sasum;
pub use bindings::rocblas_dasum;
pub use bindings::rocblas_scasum;
pub use bindings::rocblas_dzasum;

pub use bindings::rocblas_sasum_batched;
pub use bindings::rocblas_dasum_batched;
pub use bindings::rocblas_scasum_batched;
pub use bindings::rocblas_dzasum_batched;

pub use bindings::rocblas_sasum_strided_batched;
pub use bindings::rocblas_dasum_strided_batched;
pub use bindings::rocblas_scasum_strided_batched;
pub use bindings::rocblas_dzasum_strided_batched;

pub use bindings::rocblas_snrm2;
pub use bindings::rocblas_dnrm2;
pub use bindings::rocblas_scnrm2;
pub use bindings::rocblas_dznrm2;

pub use bindings::rocblas_snrm2_batched;
pub use bindings::rocblas_dnrm2_batched;
pub use bindings::rocblas_scnrm2_batched;
pub use bindings::rocblas_dznrm2_batched;

pub use bindings::rocblas_snrm2_strided_batched;
pub use bindings::rocblas_dnrm2_strided_batched;
pub use bindings::rocblas_scnrm2_strided_batched;
pub use bindings::rocblas_dznrm2_strided_batched;

pub use bindings::rocblas_isamax;
pub use bindings::rocblas_idamax;
pub use bindings::rocblas_icamax;
pub use bindings::rocblas_izamax;

pub use bindings::rocblas_isamax_batched;
pub use bindings::rocblas_idamax_batched;
pub use bindings::rocblas_icamax_batched;
pub use bindings::rocblas_izamax_batched;

pub use bindings::rocblas_isamax_strided_batched;
pub use bindings::rocblas_idamax_strided_batched;
pub use bindings::rocblas_icamax_strided_batched;
pub use bindings::rocblas_izamax_strided_batched;

pub use bindings::rocblas_isamin;
pub use bindings::rocblas_idamin;
pub use bindings::rocblas_icamin;
pub use bindings::rocblas_izamin;

pub use bindings::rocblas_isamin_batched;
pub use bindings::rocblas_idamin_batched;
pub use bindings::rocblas_icamin_batched;
pub use bindings::rocblas_izamin_batched;

pub use bindings::rocblas_isamin_strided_batched;
pub use bindings::rocblas_idamin_strided_batched;
pub use bindings::rocblas_icamin_strided_batched;
pub use bindings::rocblas_izamin_strided_batched;

pub use bindings::rocblas_sswap;
pub use bindings::rocblas_dswap;
pub use bindings::rocblas_cswap;
pub use bindings::rocblas_zswap;

pub use bindings::rocblas_sswap_batched;
pub use bindings::rocblas_dswap_batched;
pub use bindings::rocblas_cswap_batched;
pub use bindings::rocblas_zswap_batched;

pub use bindings::rocblas_sswap_strided_batched;
pub use bindings::rocblas_dswap_strided_batched;
pub use bindings::rocblas_cswap_strided_batched;
pub use bindings::rocblas_zswap_strided_batched;

pub use bindings::rocblas_srot;
pub use bindings::rocblas_drot;
pub use bindings::rocblas_crot;
pub use bindings::rocblas_csrot;
pub use bindings::rocblas_zrot;
pub use bindings::rocblas_zdrot;

pub use bindings::rocblas_srot_batched;
pub use bindings::rocblas_drot_batched;
pub use bindings::rocblas_crot_batched;
pub use bindings::rocblas_csrot_batched;
pub use bindings::rocblas_zrot_batched;
pub use bindings::rocblas_zdrot_batched;

pub use bindings::rocblas_srot_strided_batched;
pub use bindings::rocblas_drot_strided_batched;
pub use bindings::rocblas_crot_strided_batched;
pub use bindings::rocblas_csrot_strided_batched;
pub use bindings::rocblas_zrot_strided_batched;
pub use bindings::rocblas_zdrot_strided_batched;

pub use bindings::rocblas_srotg;
pub use bindings::rocblas_drotg;
pub use bindings::rocblas_crotg;
pub use bindings::rocblas_zrotg;

pub use bindings::rocblas_srotg_batched;
pub use bindings::rocblas_drotg_batched;
pub use bindings::rocblas_crotg_batched;
pub use bindings::rocblas_zrotg_batched;

pub use bindings::rocblas_srotg_strided_batched;
pub use bindings::rocblas_drotg_strided_batched;
pub use bindings::rocblas_crotg_strided_batched;
pub use bindings::rocblas_zrotg_strided_batched;

pub use bindings::rocblas_srotm;
pub use bindings::rocblas_drotm;

pub use bindings::rocblas_srotm_batched;
pub use bindings::rocblas_drotm_batched;

pub use bindings::rocblas_srotm_strided_batched;
pub use bindings::rocblas_drotm_strided_batched;

pub use bindings::rocblas_srotmg;
pub use bindings::rocblas_drotmg;

pub use bindings::rocblas_srotmg_batched;
pub use bindings::rocblas_drotmg_batched;

pub use bindings::rocblas_srotmg_strided_batched;
pub use bindings::rocblas_drotmg_strided_batched;

// Level 2 BLAS
pub use bindings::rocblas_sgemv;
pub use bindings::rocblas_dgemv;
pub use bindings::rocblas_cgemv;
pub use bindings::rocblas_zgemv;

pub use bindings::rocblas_sgemv_batched;
pub use bindings::rocblas_dgemv_batched;
pub use bindings::rocblas_cgemv_batched;
pub use bindings::rocblas_zgemv_batched;
pub use bindings::rocblas_hshgemv_batched;
pub use bindings::rocblas_hssgemv_batched;
pub use bindings::rocblas_tstgemv_batched;
pub use bindings::rocblas_tssgemv_batched;

pub use bindings::rocblas_sgemv_strided_batched;
pub use bindings::rocblas_dgemv_strided_batched;
pub use bindings::rocblas_cgemv_strided_batched;
pub use bindings::rocblas_zgemv_strided_batched;
pub use bindings::rocblas_hshgemv_strided_batched;
pub use bindings::rocblas_hssgemv_strided_batched;
pub use bindings::rocblas_tstgemv_strided_batched;
pub use bindings::rocblas_tssgemv_strided_batched;

pub use bindings::rocblas_sgbmv;
pub use bindings::rocblas_dgbmv;
pub use bindings::rocblas_cgbmv;
pub use bindings::rocblas_zgbmv;

pub use bindings::rocblas_sgbmv_batched;
pub use bindings::rocblas_dgbmv_batched;
pub use bindings::rocblas_cgbmv_batched;
pub use bindings::rocblas_zgbmv_batched;

pub use bindings::rocblas_sgbmv_strided_batched;
pub use bindings::rocblas_dgbmv_strided_batched;
pub use bindings::rocblas_cgbmv_strided_batched;
pub use bindings::rocblas_zgbmv_strided_batched;

pub use bindings::rocblas_chbmv;
pub use bindings::rocblas_zhbmv;

// Level 3 BLAS
pub use bindings::rocblas_sgemm;
pub use bindings::rocblas_dgemm;
pub use bindings::rocblas_cgemm;
pub use bindings::rocblas_zgemm;

pub use bindings::rocblas_sgemm_batched;
pub use bindings::rocblas_dgemm_batched;
pub use bindings::rocblas_cgemm_batched;
pub use bindings::rocblas_zgemm_batched;

pub use bindings::rocblas_sgemm_strided_batched;
pub use bindings::rocblas_dgemm_strided_batched;
pub use bindings::rocblas_cgemm_strided_batched;
pub use bindings::rocblas_zgemm_strided_batched;
pub use bindings::rocblas_zhemv_strided_batched;
pub use bindings::rocblas_chbmv_batched;
pub use bindings::rocblas_zhbmv_batched;
pub use bindings::rocblas_chbmv_strided_batched;
pub use bindings::rocblas_zhbmv_strided_batched;
pub use bindings::rocblas_chemv;
pub use bindings::rocblas_zhemv;
pub use bindings::rocblas_chemv_strided_batched;
pub use bindings::rocblas_zhemv_batched;
pub use bindings::rocblas_chemv_batched;
pub use bindings::rocblas_sger;
pub use bindings::rocblas_dger;
pub use bindings::rocblas_cgeru;
pub use bindings::rocblas_zgeru;
pub use bindings::rocblas_cgerc;
pub use bindings::rocblas_zgerc;
pub use bindings::rocblas_sspr2_strided_batched;
pub use bindings::rocblas_dspr2_strided_batched;
pub use bindings::rocblas_ssyr2_batched;
pub use bindings::rocblas_csyr2_strided_batched;
pub use bindings::rocblas_zsyr2_strided_batched;
pub use bindings::rocblas_sger_strided_batched;
pub use bindings::rocblas_dger_strided_batched;
pub use bindings::rocblas_sger_batched;
pub use bindings::rocblas_dger_batched;
pub use bindings::rocblas_cgeru_batched;
pub use bindings::rocblas_zgeru_batched;
pub use bindings::rocblas_cgerc_batched;
pub use bindings::rocblas_zgerc_batched;
pub use bindings::rocblas_cgeru_strided_batched;
pub use bindings::rocblas_zgeru_strided_batched;
pub use bindings::rocblas_cgerc_strided_batched;
pub use bindings::rocblas_zgerc_strided_batched;
pub use bindings::rocblas_sspr_batched;
pub use bindings::rocblas_dspr_batched;
pub use bindings::rocblas_cspr_batched;
pub use bindings::rocblas_zspr_batched;
pub use bindings::rocblas_sspr_strided_batched;
pub use bindings::rocblas_dspr_strided_batched;
pub use bindings::rocblas_cspr_strided_batched;
pub use bindings::rocblas_dspr2_batched;
pub use bindings::rocblas_csyr2_batched;
pub use bindings::rocblas_zsyr2_batched;
pub use bindings::rocblas_ssyr2_strided_batched;
pub use bindings::rocblas_dsyr2_strided_batched;
pub use bindings::rocblas_dsyr2_batched;
pub use bindings::rocblas_sspr2_batched;
pub use bindings::rocblas_zspr_strided_batched;
pub use bindings::rocblas_dspr;
pub use bindings::rocblas_chemm;
pub use bindings::rocblas_zhemm;
pub use bindings::rocblas_cherk;
pub use bindings::rocblas_zherk;
pub use bindings::rocblas_sspr;
pub use bindings::rocblas_cspr;
pub use bindings::rocblas_zspr;
pub use bindings::rocblas_sspr2;
pub use bindings::rocblas_dspr2;
pub use bindings::rocblas_ssyr;
pub use bindings::rocblas_ssyr2;
pub use bindings::rocblas_dsyr2;
pub use bindings::rocblas_csyr;
pub use bindings::rocblas_zsyr;
pub use bindings::rocblas_csyr2;
pub use bindings::rocblas_zsyr2;
pub use bindings::rocblas_ssyr_batched;
pub use bindings::rocblas_ssyr_strided_batched;
pub use bindings::rocblas_chemm_batched;
pub use bindings::rocblas_zhemm_batched;
pub use bindings::rocblas_chemm_strided_batched;
pub use bindings::rocblas_zhemm_strided_batched;
pub use bindings::rocblas_cherk_batched;
pub use bindings::rocblas_zherk_batched;
pub use bindings::rocblas_cherk_strided_batched;
pub use bindings::rocblas_zherk_strided_batched;
pub use bindings::rocblas_cher2k;
pub use bindings::rocblas_zher2k;
pub use bindings::rocblas_cherkx;
pub use bindings::rocblas_zherkx;
pub use bindings::rocblas_cherkx_batched;
pub use bindings::rocblas_zherkx_batched;
pub use bindings::rocblas_cherkx_strided_batched;
pub use bindings::rocblas_zherkx_strided_batched;

pub use bindings::rocblas_gemm_ex;
pub use bindings::hipStream_t;
pub use bindings::rocblas_status_to_string;
pub use bindings::rocblas_initialize;
pub use bindings::rocblas_get_version_string_size;
pub use bindings::rocblas_get_version_string;
pub use bindings::rocblas_start_device_memory_size_query;
pub use bindings::rocblas_stop_device_memory_size_query;
pub use bindings::rocblas_is_device_memory_size_query;
pub use bindings::rocblas_get_device_memory_size;
pub use bindings::rocblas_set_device_memory_size;
pub use bindings::rocblas_set_workspace;
pub use bindings::rocblas_is_managing_device_memory;
pub use bindings::rocblas_is_user_managing_device_memory;
pub use bindings::rocblas_device_malloc_set_default_memory_size;
pub use bindings::rocblas_abort;


pub use bindings::rocblas_set_vector_async;
pub use bindings::rocblas_set_vector_async_64;
pub use bindings::rocblas_get_vector_async;
pub use bindings::rocblas_get_vector_async_64;
pub use bindings::rocblas_set_matrix_async;
pub use bindings::rocblas_set_matrix_async_64;
pub use bindings::rocblas_get_matrix_async;
pub use bindings::rocblas_get_matrix_async_64;
pub use bindings::rocblas_set_start_stop_events;