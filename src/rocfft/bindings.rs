/* automatically generated by rust-bindgen 0.71.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocfft_plan_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to plan structure\n  @details This type is used to declare a plan handle that can be initialized\n with ::rocfft_plan_create."]
pub type rocfft_plan = *mut rocfft_plan_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocfft_plan_description_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to plan description structure\n  @details This type is used to declare a plan description handle that can be\n initialized with ::rocfft_plan_description_create."]
pub type rocfft_plan_description = *mut rocfft_plan_description_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocfft_execution_info_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to execution info structure\n  @details This type is used to declare an execution info handle that can be\n initialized with ::rocfft_execution_info_create."]
pub type rocfft_execution_info = *mut rocfft_execution_info_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocfft_field_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to a rocFFT field structure.\n\n  @details rocFFT fields are used to hold data decomposition information which is then passed to a\n  \\ref rocfft_plan via a \\ref rocfft_plan_description\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
pub type rocfft_field = *mut rocfft_field_t;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocfft_brick_t {
    _unused: [u8; 0],
}
#[doc = " @brief Pointer type to a rocFFT brick structure.\n\n  @details rocFFT bricks are used to describe the data decomposition of fields.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
pub type rocfft_brick = *mut rocfft_brick_t;
pub const rocfft_status_e_rocfft_status_success: rocfft_status_e = 0;
pub const rocfft_status_e_rocfft_status_failure: rocfft_status_e = 1;
pub const rocfft_status_e_rocfft_status_invalid_arg_value: rocfft_status_e = 2;
pub const rocfft_status_e_rocfft_status_invalid_dimensions: rocfft_status_e = 3;
pub const rocfft_status_e_rocfft_status_invalid_array_type: rocfft_status_e = 4;
pub const rocfft_status_e_rocfft_status_invalid_strides: rocfft_status_e = 5;
pub const rocfft_status_e_rocfft_status_invalid_distance: rocfft_status_e = 6;
pub const rocfft_status_e_rocfft_status_invalid_offset: rocfft_status_e = 7;
pub const rocfft_status_e_rocfft_status_invalid_work_buffer: rocfft_status_e = 8;
#[doc = " @brief rocFFT status/error codes"]
pub type rocfft_status_e = ::std::os::raw::c_uint;
#[doc = " @brief rocFFT status/error codes"]
pub use self::rocfft_status_e as rocfft_status;
pub const rocfft_transform_type_e_rocfft_transform_type_complex_forward: rocfft_transform_type_e =
    0;
pub const rocfft_transform_type_e_rocfft_transform_type_complex_inverse: rocfft_transform_type_e =
    1;
pub const rocfft_transform_type_e_rocfft_transform_type_real_forward: rocfft_transform_type_e = 2;
pub const rocfft_transform_type_e_rocfft_transform_type_real_inverse: rocfft_transform_type_e = 3;
#[doc = " @brief Type of transform"]
pub type rocfft_transform_type_e = ::std::os::raw::c_uint;
#[doc = " @brief Type of transform"]
pub use self::rocfft_transform_type_e as rocfft_transform_type;
pub const rocfft_precision_e_rocfft_precision_single: rocfft_precision_e = 0;
pub const rocfft_precision_e_rocfft_precision_double: rocfft_precision_e = 1;
pub const rocfft_precision_e_rocfft_precision_half: rocfft_precision_e = 2;
#[doc = " @brief Precision"]
pub type rocfft_precision_e = ::std::os::raw::c_uint;
#[doc = " @brief Precision"]
pub use self::rocfft_precision_e as rocfft_precision;
pub const rocfft_result_placement_e_rocfft_placement_inplace: rocfft_result_placement_e = 0;
pub const rocfft_result_placement_e_rocfft_placement_notinplace: rocfft_result_placement_e = 1;
#[doc = " @brief Result placement\n  @details Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place."]
pub type rocfft_result_placement_e = ::std::os::raw::c_uint;
#[doc = " @brief Result placement\n  @details Declares where the output of the transform should be\n  placed.  Note that input buffers may still be overwritten\n  during execution of a transform, even if the transform is not\n  in-place."]
pub use self::rocfft_result_placement_e as rocfft_result_placement;
pub const rocfft_array_type_e_rocfft_array_type_complex_interleaved: rocfft_array_type_e = 0;
pub const rocfft_array_type_e_rocfft_array_type_complex_planar: rocfft_array_type_e = 1;
pub const rocfft_array_type_e_rocfft_array_type_real: rocfft_array_type_e = 2;
pub const rocfft_array_type_e_rocfft_array_type_hermitian_interleaved: rocfft_array_type_e = 3;
pub const rocfft_array_type_e_rocfft_array_type_hermitian_planar: rocfft_array_type_e = 4;
pub const rocfft_array_type_e_rocfft_array_type_unset: rocfft_array_type_e = 5;
#[doc = " @brief Array type"]
pub type rocfft_array_type_e = ::std::os::raw::c_uint;
#[doc = " @brief Array type"]
pub use self::rocfft_array_type_e as rocfft_array_type;
pub const rocfft_comm_type_e_rocfft_comm_none: rocfft_comm_type_e = 0;
pub const rocfft_comm_type_e_rocfft_comm_mpi: rocfft_comm_type_e = 1;
#[doc = " @brief Communicator type for distributed transforms"]
pub type rocfft_comm_type_e = ::std::os::raw::c_uint;
#[doc = " @brief Communicator type for distributed transforms"]
pub use self::rocfft_comm_type_e as rocfft_comm_type;
unsafe extern "C" {
    #[doc = " @brief Library setup function, called once in program before start of\n library use"]
    pub fn rocfft_setup() -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Library cleanup function, called once in program after end of library\n use"]
    pub fn rocfft_cleanup() -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Create an FFT plan\n\n  @details This API creates a plan, which the user can execute\n  subsequently.  This function takes many of the fundamental\n  parameters needed to specify a transform.\n\n  The dimensions parameter can take a value of 1, 2, or 3. The\n  'lengths' array specifies the size of data in each dimension. Note\n  that lengths[0] is the size of the innermost dimension, lengths[1]\n  is the next higher dimension and so on (column-major ordering).\n\n  The 'number_of_transforms' parameter specifies how many\n  transforms (of the same kind) needs to be computed. By specifying\n  a value greater than 1, a batch of transforms can be computed\n  with a single API call.\n\n  Additionally, a handle to a plan description can be passed for\n  more detailed transforms. For simple transforms, this parameter\n  can be set to NULL.\n\n  The plan must be destroyed with a call to ::rocfft_plan_destroy.\n\n  @param[out] plan plan handle\n  @param[in] placement placement of result\n  @param[in] transform_type type of transform\n  @param[in] precision precision\n  @param[in] dimensions dimensions\n  @param[in] lengths dimensions-sized array of transform lengths\n  @param[in] number_of_transforms number of transforms\n  @param[in] description description handle created by\n rocfft_plan_description_create; can be\n  NULL for simple transforms"]
    pub fn rocfft_plan_create(
        plan: *mut rocfft_plan,
        placement: rocfft_result_placement,
        transform_type: rocfft_transform_type,
        precision: rocfft_precision,
        dimensions: usize,
        lengths: *const usize,
        number_of_transforms: usize,
        description: rocfft_plan_description,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Execute an FFT plan\n\n  @details This API executes an FFT plan on buffers given by the user.\n\n  If the transform is in-place, only the input buffer is needed and\n  the output buffer parameter can be set to NULL. For not in-place\n  transforms, output buffers have to be specified.\n\n  Input and output buffers are arrays of pointers.  Interleaved\n  array formats are the default, and require just one pointer per\n  input or output buffer.  Planar array formats require two\n  pointers per input or output buffer - real and imaginary\n  pointers, in that order.\n\n  If fields have been set for transform input or output, these\n  arrays have one pointer per brick in the input or output field,\n  provided in the order that the bricks were added to the field.\n\n  Note that input buffers may still be overwritten during execution\n  of a transform, even if the transform is not in-place.\n\n  The final parameter in this function is a rocfft_execution_info\n  handle. This optional parameter serves as a way for the user to control\n  execution streams and work buffers.\n\n  @param[in] plan plan handle\n  @param[in,out] in_buffer array (of size 1 for interleaved data, of size 2\n for planar data, or one per brick if an input field is set) of input buffers\n  @param[in,out] out_buffer array (of size 1 for interleaved data, of size 2\n for planar data, or one per brick if an output field is set) of output buffers,\n ignored for in-place transforms\n  @param[in] info execution info handle created by\n rocfft_execution_info_create"]
    pub fn rocfft_execute(
        plan: rocfft_plan,
        in_buffer: *mut *mut ::std::os::raw::c_void,
        out_buffer: *mut *mut ::std::os::raw::c_void,
        info: rocfft_execution_info,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Destroy an FFT plan\n  @details This API frees the plan after it is no longer needed.\n  @param[in] plan plan handle"]
    pub fn rocfft_plan_destroy(plan: rocfft_plan) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set scaling factor.\n  @details rocFFT multiplies each element of the result by the given factor at the end of the transform.\n\n  The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.\n\n  @param[in] description description handle\n  @param[in] scale_factor scaling factor"]
    pub fn rocfft_plan_description_set_scale_factor(
        description: rocfft_plan_description,
        scale_factor: f64,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = "  @brief Set advanced data layout parameters on a plan description\n\n  @details This API specifies advanced layout of input/output\n  buffers for a plan description.\n\n  The following parameters are supported for inputs and outputs:\n\n  * Array type (real, hermitian, or complex data, in either\n    interleaved or planar format).\n      * Real forward transforms require real input and hermitian output.\n      * Real inverse transforms require hermitian input and real output.\n      * Complex transforms require complex input and output.\n      * Hermitian and complex data defaults to interleaved if a specific\nformat is not specified.\n  * Offset of first data element in the data buffer.  Defaults to 0 if unspecified.\n  * Stride between consecutive elements in each dimension.  Defaults\nto contiguous data in all dimensions if unspecified.\n  * Distance between consecutive batches.  Defaults to contiguous\nbatches if unspecified.\n\n  Not all combinations of array types are supported and error codes\n  will be returned for unsupported cases.\n\n  Offset, stride, and distance for either input or output provided\n  here is ignored if a field is set for the corresponding input or\n  output.\n\n  @param[in, out] description description handle\n  @param[in] in_array_type array type of input buffer\n  @param[in] out_array_type array type of output buffer\n  @param[in] in_offsets offsets, in element units, to start of data in input buffer\n  @param[in] out_offsets offsets, in element units, to start of data in output buffer\n  @param[in] in_strides_size size of in_strides array (must be equal to transform dimensions)\n  @param[in] in_strides array of strides, in each dimension, of\n   input buffer; if set to null ptr library chooses defaults\n  @param[in] in_distance distance between start of each data instance in input buffer\n  @param[in] out_strides_size size of out_strides array (must be\n  equal to transform dimensions)\n  @param[in] out_strides array of strides, in each dimension, of\n   output buffer; if set to null ptr library chooses defaults\n  @param[in] out_distance distance between start of each data instance in output buffer"]
    pub fn rocfft_plan_description_set_data_layout(
        description: rocfft_plan_description,
        in_array_type: rocfft_array_type,
        out_array_type: rocfft_array_type,
        in_offsets: *const usize,
        out_offsets: *const usize,
        in_strides_size: usize,
        in_strides: *const usize,
        in_distance: usize,
        out_strides_size: usize,
        out_strides: *const usize,
        out_distance: usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Create a rocfft field struct.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_field_create(field: *mut rocfft_field) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Destroy a rocfft field struct\n\n The field struct can be destroyed after being added to the plan description; it is not used for\n plan execution.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_field_destroy(field: rocfft_field) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Get library version string\n\n @param[in, out] buf buffer that receives the version string\n @param[in] len length of buf, minimum 30 characters"]
    pub fn rocfft_get_version_string(buf: *mut ::std::os::raw::c_char, len: usize)
        -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set the communication library for distributed transforms.\n\n  @details Set the multi-processing communication library for a plan.\n\n  Multi-processing communication libraries require library-specific\n  handle to also be specified.  For MPI libraries, this is a\n  pointer to an MPI communicator.\n\n  @param[in] description description handle\n  @param[in] comm_type communicator type\n  @param[in] comm_handle handle to communication-library-specific state\n"]
    pub fn rocfft_plan_description_set_comm(
        description: rocfft_plan_description,
        comm_type: rocfft_comm_type,
        comm_handle: *mut ::std::os::raw::c_void,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Define a brick as part of a decomposition of a field.\n\n Fields can contain a full-dimensional data distribution.  The\n decomposition is specified by providing a lower coordinate and an\n upper coordinate in the field's index space.  The lower coordinate\n is inclusive (contained within the brick) and the upper coordinate\n is exclusive (first index past the end of the brick).\n\n One must also provide a stride for the brick data which specifies\n how the brick's data is arranged in memory.\n\n All coordinates and strides must include batch dimensions, and are in\n column-major order (fastest-moving dimension first).\n\n A HIP device ID is also provided - each brick may reside on a\n different device.\n\n All arrays may be re-used or freed immediately after the function returns.\n\n @param[out] brick: brick structure\n @param[in] field_lower: array of length dim specifying the lower index (inclusive) for the brick in the\n field's index space.\n @param[in] field_upper: array of length dim specifying the upper index (exclusive) for the brick in the\n field's index space.\n @param[in] brick_stride: array of length dim specifying the brick's stride in memory\n @param[in] dim_with_batch length of the arrays; this must match the dimension of\n the FFT plus one for the batch dimension.\n @param[in] deviceID: HIP device ID for the device on which the brick's data is resident.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_brick_create(
        brick: *mut rocfft_brick,
        field_lower: *const usize,
        field_upper: *const usize,
        brick_stride: *const usize,
        dim_with_batch: usize,
        deviceID: ::std::os::raw::c_int,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Deallocate a brick created with rocfft_brick_create.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_brick_destroy(brick: rocfft_brick) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Add a brick to a field.\n\n Note that the order in which the bricks are added is significant;\n the pointers provided for each brick to ::rocfft_execute are in\n the same order that the bricks were added to the field.\n\n The brick may be added to another field or destroyed any time\n after this function returns.\n\n @param[in, out] field: \\ref rocfft_field struct which holds the brick decomposition.\n @param[in] brick: \\ref rocfft_brick struct to add to the field.\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_field_add_brick(field: rocfft_field, brick: rocfft_brick) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Add a \\ref rocfft_field to a \\ref rocfft_plan_description as an input.\n\n The field may be reused or freed immediately after the function returns.\n\n @param[in, out] description: \\ref rocfft_plan_description that will pass the field information to plan creation\n @param[in] field: \\ref rocfft_field struct added as an input field\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_plan_description_add_infield(
        description: rocfft_plan_description,
        field: rocfft_field,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Add a \\ref rocfft_field to a \\ref rocfft_plan_description as an output.\n\n The field may be reused or freed immediately after the function returns.\n\n @param[in, out] description: \\ref rocfft_plan_description  that will pass the field information to plan creation\n @param[in] field: \\ref rocfft_field struct added as an output field\n\n  @warning Experimental!  This feature is part of an experimental API preview."]
    pub fn rocfft_plan_description_add_outfield(
        description: rocfft_plan_description,
        field: rocfft_field,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Get work buffer size\n  @details Get the work buffer size required for a plan.\n  @param[in] plan plan handle\n  @param[out] size_in_bytes size of needed work buffer in bytes"]
    pub fn rocfft_plan_get_work_buffer_size(
        plan: rocfft_plan,
        size_in_bytes: *mut usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Print all plan information\n  @details Prints plan details to stdout, to aid debugging\n  @param[in] plan plan handle"]
    pub fn rocfft_plan_get_print(plan: rocfft_plan) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Create plan description\n  @details This API creates a plan description with which the user\n can set extra plan properties.  The plan description must be freed\n with a call to ::rocfft_plan_description_destroy.\n  @param[out] description plan description handle"]
    pub fn rocfft_plan_description_create(
        description: *mut rocfft_plan_description,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Destroy a plan description\n  @details This API frees the plan description.  A plan description\n  can be freed any time after it is passed to ::rocfft_plan_create.\n  @param[in] description plan description handle"]
    pub fn rocfft_plan_description_destroy(description: rocfft_plan_description) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Create execution info\n  @details This API creates an execution info with which the user\n can control plan execution and work buffers.  The execution info must be freed\n with a call to ::rocfft_execution_info_destroy.\n  @param[out] info execution info handle"]
    pub fn rocfft_execution_info_create(info: *mut rocfft_execution_info) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Destroy an execution info\n  @details This API frees the execution info.  An execution info\n  object can be freed any time after it is passed to\n  ::rocfft_execute.\n  @param[in] info execution info handle"]
    pub fn rocfft_execution_info_destroy(info: rocfft_execution_info) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set work buffer in execution info\n\n  @details This is one of the execution info functions to specify\n  optional additional information to control execution.  This API\n  provides a work buffer for the transform. It must be called\n  before ::rocfft_execute.\n\n  When a non-zero value is obtained from\n  ::rocfft_plan_get_work_buffer_size, that means the library needs a\n  work buffer to compute the transform. In this case, the user\n  should allocate the work buffer and pass it to the library via\n  this API.\n\n  If a work buffer is required for the transform but is not\n  specified using this function, ::rocfft_execute will automatically\n  allocate the required buffer and free it when execution is\n  finished.\n\n  Users should allocate their own work buffers if they need precise\n  control over the lifetimes of those buffers, or if multiple plans\n  need to share the same buffer.\n\n  @param[in] info execution info handle\n  @param[in] work_buffer work buffer\n  @param[in] size_in_bytes size of work buffer in bytes"]
    pub fn rocfft_execution_info_set_work_buffer(
        info: rocfft_execution_info,
        work_buffer: *mut ::std::os::raw::c_void,
        size_in_bytes: usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set stream in execution info\n  @details Associates an existing compute stream to a plan.  This\n must be called before the call to ::rocfft_execute.\n\n  Once the association is made, execution of the FFT will run the\n  computation through the specified stream.\n\n  The stream must be of type hipStream_t. It is an error to pass\n  the address of a hipStream_t object.\n\n  @param[in] info execution info handle\n  @param[in] stream underlying compute stream"]
    pub fn rocfft_execution_info_set_stream(
        info: rocfft_execution_info,
        stream: *mut ::std::os::raw::c_void,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set a load callback for a plan execution (experimental)\n  @details This function specifies a user-defined callback function\n  that is run to load input from global memory at the start of the\n  transform.  Callbacks are an experimental feature in rocFFT.\n\n  Callback function pointers/data are given as arrays, with one\n  function/data pointer per device executing this plan.  Currently,\n  plans can only use one device.\n\n  The provided function pointers replace any previously-specified\n  load callback for this execution info handle.\n\n  Load callbacks have the following signature:\n\n  @code\n  T load_cb(T* data, size_t offset, void* cbdata, void* sharedMem);\n  @endcode\n\n  'T' is the type of a single element of the input buffer.  It is\n  the caller's responsibility to ensure that the function type is\n  appropriate for the plan (for example, a single-precision\n  real-to-complex transform would load single-precision real\n  elements).\n\n  A null value for 'cb' may be specified to clear any previously\n  registered load callback.\n\n  Currently, 'shared_mem_bytes' must be 0.  Callbacks are not\n  supported on transforms that use planar formats for either input\n  or output.\n\n  @param[in] info execution info handle\n  @param[in] cb_functions callback function pointers\n  @param[in] cb_data callback function data, passed to the function pointer when it is called\n  @param[in] shared_mem_bytes amount of shared memory to allocate for the callback function to use"]
    pub fn rocfft_execution_info_set_load_callback(
        info: rocfft_execution_info,
        cb_functions: *mut *mut ::std::os::raw::c_void,
        cb_data: *mut *mut ::std::os::raw::c_void,
        shared_mem_bytes: usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Set a store callback for a plan execution (experimental)\n  @details This function specifies a user-defined callback function\n  that is run to store output to global memory at the end of the\n  transform.  Callbacks are an experimental feature in rocFFT.\n\n  Callback function pointers/data are given as arrays, with one\n  function/data pointer per device executing this plan.  Currently,\n  plans can only use one device.\n\n  The provided function pointers replace any previously-specified\n  store callback for this execution info handle.\n\n  Store callbacks have the following signature:\n\n  @code\n  void store_cb(T* data, size_t offset, T element, void* cbdata, void* sharedMem);\n  @endcode\n\n  'T' is the type of a single element of the output buffer.  It is\n  the caller's responsibility to ensure that the function type is\n  appropriate for the plan (for example, a single-precision\n  real-to-complex transform would store single-precision complex\n  elements).\n\n  A null value for 'cb' may be specified to clear any previously\n  registered store callback.\n\n  Currently, 'shared_mem_bytes' must be 0.  Callbacks are not\n  supported on transforms that use planar formats for either input\n  or output.\n\n  @param[in] info execution info handle\n  @param[in] cb_functions callbacks function pointers\n  @param[in] cb_data callback function data, passed to the function pointer when it is called\n  @param[in] shared_mem_bytes amount of shared memory to allocate for the callback function to use"]
    pub fn rocfft_execution_info_set_store_callback(
        info: rocfft_execution_info,
        cb_functions: *mut *mut ::std::os::raw::c_void,
        cb_data: *mut *mut ::std::os::raw::c_void,
        shared_mem_bytes: usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Serialize compiled kernel cache\n\n  @details Serialize rocFFT's cache of compiled kernels into a\n  buffer.  This buffer is allocated by rocFFT and must be freed\n  with a call to ::rocfft_cache_buffer_free.  The length of the\n  buffer in bytes is written to 'buffer_len_bytes'."]
    pub fn rocfft_cache_serialize(
        buffer: *mut *mut ::std::os::raw::c_void,
        buffer_len_bytes: *mut usize,
    ) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Free cache serialization buffer\n\n  @details Deallocate a buffer allocated by ::rocfft_cache_serialize."]
    pub fn rocfft_cache_buffer_free(buffer: *mut ::std::os::raw::c_void) -> rocfft_status;
}
unsafe extern "C" {
    #[doc = " @brief Deserialize a buffer into the compiled kernel cache.\n\n  @details Kernels in the buffer that match already-cached kernels\n  will replace those kernels that are in the cache.  Already-cached\n  kernels that do not match those in the buffer are unmodified by\n  this operation.  The cache is unmodified if either a null buffer\n  pointer or a zero length is passed."]
    pub fn rocfft_cache_deserialize(
        buffer: *const ::std::os::raw::c_void,
        buffer_len_bytes: usize,
    ) -> rocfft_status;
}
