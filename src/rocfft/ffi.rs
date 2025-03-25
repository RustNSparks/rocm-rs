// FFI module for rocFFT
// This file re-exports the necessary symbols from the auto-generated bindings

// Import the raw bindings from the auto-generated module
use crate::rocfft::bindings;

// Re-export the necessary types, constants, and functions

// Types
pub use bindings::{
    // Handle types
    rocfft_plan,
    rocfft_plan_description,
    rocfft_execution_info,
    rocfft_field,
    rocfft_brick,

    // Status type
    rocfft_status_e as rocfft_status_t,

    // Complex types
    rocfft_status as rocfft_status_t_alias,
    rocfft_transform_type as rocfft_transform_type_t_alias,
    rocfft_precision as rocfft_precision_t_alias,
    rocfft_result_placement as rocfft_result_placement_t_alias,
    rocfft_array_type as rocfft_array_type_t_alias,
    rocfft_comm_type as rocfft_comm_type_t_alias,
};

// Status constants
pub use bindings::{
    rocfft_status_e_rocfft_status_success as STATUS_SUCCESS,
    rocfft_status_e_rocfft_status_failure as STATUS_FAILURE,
    rocfft_status_e_rocfft_status_invalid_arg_value as STATUS_INVALID_ARG_VALUE,
    rocfft_status_e_rocfft_status_invalid_dimensions as STATUS_INVALID_DIMENSIONS,
    rocfft_status_e_rocfft_status_invalid_array_type as STATUS_INVALID_ARRAY_TYPE,
    rocfft_status_e_rocfft_status_invalid_strides as STATUS_INVALID_STRIDES,
    rocfft_status_e_rocfft_status_invalid_distance as STATUS_INVALID_DISTANCE,
    rocfft_status_e_rocfft_status_invalid_offset as STATUS_INVALID_OFFSET,
    rocfft_status_e_rocfft_status_invalid_work_buffer as STATUS_INVALID_WORK_BUFFER,
};

// Transform type constants
pub use bindings::{
    rocfft_transform_type_e_rocfft_transform_type_complex_forward as TRANSFORM_TYPE_COMPLEX_FORWARD,
    rocfft_transform_type_e_rocfft_transform_type_complex_inverse as TRANSFORM_TYPE_COMPLEX_INVERSE,
    rocfft_transform_type_e_rocfft_transform_type_real_forward as TRANSFORM_TYPE_REAL_FORWARD,
    rocfft_transform_type_e_rocfft_transform_type_real_inverse as TRANSFORM_TYPE_REAL_INVERSE,
};

// Precision constants
pub use bindings::{
    rocfft_precision_e_rocfft_precision_single as PRECISION_SINGLE,
    rocfft_precision_e_rocfft_precision_double as PRECISION_DOUBLE,
    rocfft_precision_e_rocfft_precision_half as PRECISION_HALF,
};

// Placement constants
pub use bindings::{
    rocfft_result_placement_e_rocfft_placement_inplace as PLACEMENT_INPLACE,
    rocfft_result_placement_e_rocfft_placement_notinplace as PLACEMENT_NOTINPLACE,
};

// Array type constants
pub use bindings::{
    rocfft_array_type_e_rocfft_array_type_complex_interleaved as ARRAY_TYPE_COMPLEX_INTERLEAVED,
    rocfft_array_type_e_rocfft_array_type_complex_planar as ARRAY_TYPE_COMPLEX_PLANAR,
    rocfft_array_type_e_rocfft_array_type_real as ARRAY_TYPE_REAL,
    rocfft_array_type_e_rocfft_array_type_hermitian_interleaved as ARRAY_TYPE_HERMITIAN_INTERLEAVED,
    rocfft_array_type_e_rocfft_array_type_hermitian_planar as ARRAY_TYPE_HERMITIAN_PLANAR,
    rocfft_array_type_e_rocfft_array_type_unset as ARRAY_TYPE_UNSET,
};

// Communicator type constants
pub use bindings::{
    rocfft_comm_type_e_rocfft_comm_none as COMM_TYPE_NONE,
    rocfft_comm_type_e_rocfft_comm_mpi as COMM_TYPE_MPI,
};

// Function re-exports

// Library setup/cleanup
pub use bindings::{
    rocfft_setup,
    rocfft_cleanup,
    rocfft_get_version_string,
};

// Plan creation/destruction
pub use bindings::{
    rocfft_plan_create,
    rocfft_plan_destroy,
    rocfft_plan_get_work_buffer_size,
    rocfft_plan_get_print,
};

// Plan description
pub use bindings::{
    rocfft_plan_description_create,
    rocfft_plan_description_destroy,
    rocfft_plan_description_set_scale_factor,
    rocfft_plan_description_set_data_layout,
    rocfft_plan_description_set_comm,
};

// Execution
pub use bindings::{
    rocfft_execute,
    rocfft_execution_info_create,
    rocfft_execution_info_destroy,
    rocfft_execution_info_set_work_buffer,
    rocfft_execution_info_set_stream,
    rocfft_execution_info_set_load_callback,
    rocfft_execution_info_set_store_callback,
};

// Field/Brick (distributed computation)
pub use bindings::{
    rocfft_field_create,
    rocfft_field_destroy,
    rocfft_brick_create,
    rocfft_brick_destroy,
    rocfft_field_add_brick,
    rocfft_plan_description_add_infield,
    rocfft_plan_description_add_outfield,
};

// Cache management
pub use bindings::{
    rocfft_cache_serialize,
    rocfft_cache_buffer_free,
    rocfft_cache_deserialize,
};