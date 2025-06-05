// src/rocarray/kernels.rs - Complete implementation of GPU kernels for ROCArray operations
use crate::error::Result;
use crate::hip::{DeviceMemory, Dim3, Function, Module, Stream, calculate_grid_1d};
use crate::rocarray::Shape;
use std::ffi::c_void;
use std::sync::Once;

static INIT: Once = Once::new();
static mut KERNELS_MODULE: Option<Module> = None;

// Trait for types that support numeric operations
pub trait NumericOps: Copy + Default + 'static {
    const TYPE_NAME: &'static str;
}

impl NumericOps for f32 {
    const TYPE_NAME: &'static str = "float";
}

impl NumericOps for f64 {
    const TYPE_NAME: &'static str = "double";
}

impl NumericOps for i32 {
    const TYPE_NAME: &'static str = "int";
}

impl NumericOps for u32 {
    const TYPE_NAME: &'static str = "uint";
}

impl NumericOps for i64 {
    const TYPE_NAME: &'static str = "long";
}

impl NumericOps for u64 {
    const TYPE_NAME: &'static str = "ulong";
}

impl NumericOps for i16 {
    const TYPE_NAME: &'static str = "short";
}

impl NumericOps for u16 {
    const TYPE_NAME: &'static str = "ushort";
}

impl NumericOps for i8 {
    const TYPE_NAME: &'static str = "char";
}

impl NumericOps for u8 {
    const TYPE_NAME: &'static str = "uchar";
}

// Trait for transposable operations
pub trait TransposableOps: Copy + Default + 'static {
    const TYPE_NAME: &'static str;
}

impl TransposableOps for f32 {
    const TYPE_NAME: &'static str = "float";
}

impl TransposableOps for f64 {
    const TYPE_NAME: &'static str = "double";
}

impl TransposableOps for i32 {
    const TYPE_NAME: &'static str = "int";
}

impl TransposableOps for u32 {
    const TYPE_NAME: &'static str = "uint";
}

impl TransposableOps for i64 {
    const TYPE_NAME: &'static str = "long";
}

impl TransposableOps for u64 {
    const TYPE_NAME: &'static str = "ulong";
}

// Traits for other operations
pub trait Mappable<U>: Copy + Default + 'static {
    fn map_kernel_name() -> &'static str;
}

pub trait Filterable: Copy + Default + 'static {
    fn filter_kernel_name() -> &'static str;
}

pub trait Reducible: Copy + Default + 'static {
    fn reduce_kernel_name() -> &'static str;
}

pub trait Searchable: Copy + Default + 'static {
    fn search_kernel_name() -> &'static str;
}

pub trait RangeOps: Copy + Default + 'static {
    fn range_kernel_name() -> &'static str;
}

// Implement traits for basic types
macro_rules! impl_kernel_traits {
    ($($t:ty),*) => {
        $(
            impl<U: Copy + Default + 'static> Mappable<U> for $t {
                fn map_kernel_name() -> &'static str { "generic_map" }
            }

            impl Filterable for $t {
                fn filter_kernel_name() -> &'static str { "generic_filter" }
            }

            impl Reducible for $t {
                fn reduce_kernel_name() -> &'static str { "generic_reduce" }
            }

            impl Searchable for $t {
                fn search_kernel_name() -> &'static str { "generic_search" }
            }

            impl RangeOps for $t {
                fn range_kernel_name() -> &'static str {
                    match stringify!($t) {
                        "f32" => "generic_range_float",
                        "f64" => "generic_range_double",
                        "i32" => "generic_range_int",
                        "u32" => "generic_range_uint",
                        "i64" => "generic_range_long",
                        "u64" => "generic_range_ulong",
                        _ => "generic_range_float",
                    }
                }
            }
        )*
    };
}

impl_kernel_traits!(f32, f64, i32, u32, i64, u64, i16, u16, i8, u8);

// Kernel initialization
fn init_kernels() -> Result<()> {
    INIT.call_once(|| {
        let kernel_source = include_str!("kernels.hip");

        match crate::hip::compile_and_load(kernel_source, &[]) {
            Ok(module) => unsafe {
                KERNELS_MODULE = Some(module);
            },
            Err(e) => {
                eprintln!("Failed to load kernels: {:?}", e);
            }
        }
    });
    Ok(())
}

fn get_kernel_function(name: &str) -> Result<Function> {
    init_kernels()?;

    unsafe {
        if let Some(ref module) = KERNELS_MODULE {
            Ok(module.get_function(name)?)
        } else {
            Err(crate::error::Error::InvalidOperation(
                "Kernels not initialized".to_string(),
            ))
        }
    }
}

// =============================================================================
// Element-wise operations
// =============================================================================

pub fn elementwise_add<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_add_async(a, b, result, len, &Stream::new()?)
}

pub fn elementwise_add_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_add_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_sub<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_sub_async(a, b, result, len, &Stream::new()?)
}

pub fn elementwise_sub_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_sub_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_mul<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_mul_async(a, b, result, len, &Stream::new()?)
}

pub fn elementwise_mul_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_mul_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_div<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_div_async(a, b, result, len, &Stream::new()?)
}

pub fn elementwise_div_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_div_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Broadcasting operations
// =============================================================================

pub fn elementwise_add_broadcast<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_add_broadcast_async(
        a,
        b,
        result,
        a_shape,
        b_shape,
        result_shape,
        &Stream::new()?,
    )
}

pub fn elementwise_add_broadcast_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_add_broadcast_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let total_elements = result_shape.size();
    let grid_dim = calculate_grid_1d(total_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data for GPU
    let a_dims: Vec<u32> = a_shape.dims().iter().map(|&x| x as u32).collect();
    let b_dims: Vec<u32> = b_shape.dims().iter().map(|&x| x as u32).collect();
    let result_dims: Vec<u32> = result_shape.dims().iter().map(|&x| x as u32).collect();

    let a_strides: Vec<u32> = a_shape.strides().iter().map(|&x| x as u32).collect();
    let b_strides: Vec<u32> = b_shape.strides().iter().map(|&x| x as u32).collect();

    let a_ndim = a_shape.ndim() as u32;
    let b_ndim = b_shape.ndim() as u32;
    let result_ndim = result_shape.ndim() as u32;
    let total_elements_u32 = total_elements as u32;

    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        a_dims.as_ptr() as *mut c_void,
        a_strides.as_ptr() as *mut c_void,
        &a_ndim as *const u32 as *mut c_void,
        b_dims.as_ptr() as *mut c_void,
        b_strides.as_ptr() as *mut c_void,
        &b_ndim as *const u32 as *mut c_void,
        result_dims.as_ptr() as *mut c_void,
        &result_ndim as *const u32 as *mut c_void,
        &total_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_sub_broadcast<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_sub_broadcast_async(
        a,
        b,
        result,
        a_shape,
        b_shape,
        result_shape,
        &Stream::new()?,
    )
}

pub fn elementwise_sub_broadcast_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_sub_broadcast_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let total_elements = result_shape.size();
    let grid_dim = calculate_grid_1d(total_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data for GPU
    let a_dims: Vec<u32> = a_shape.dims().iter().map(|&x| x as u32).collect();
    let b_dims: Vec<u32> = b_shape.dims().iter().map(|&x| x as u32).collect();
    let result_dims: Vec<u32> = result_shape.dims().iter().map(|&x| x as u32).collect();

    let a_strides: Vec<u32> = a_shape.strides().iter().map(|&x| x as u32).collect();
    let b_strides: Vec<u32> = b_shape.strides().iter().map(|&x| x as u32).collect();

    let a_ndim = a_shape.ndim() as u32;
    let b_ndim = b_shape.ndim() as u32;
    let result_ndim = result_shape.ndim() as u32;
    let total_elements_u32 = total_elements as u32;

    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        a_dims.as_ptr() as *mut c_void,
        a_strides.as_ptr() as *mut c_void,
        &a_ndim as *const u32 as *mut c_void,
        b_dims.as_ptr() as *mut c_void,
        b_strides.as_ptr() as *mut c_void,
        &b_ndim as *const u32 as *mut c_void,
        result_dims.as_ptr() as *mut c_void,
        &result_ndim as *const u32 as *mut c_void,
        &total_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_mul_broadcast<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_mul_broadcast_async(
        a,
        b,
        result,
        a_shape,
        b_shape,
        result_shape,
        &Stream::new()?,
    )
}

pub fn elementwise_mul_broadcast_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_mul_broadcast_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let total_elements = result_shape.size();
    let grid_dim = calculate_grid_1d(total_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data for GPU
    let a_dims: Vec<u32> = a_shape.dims().iter().map(|&x| x as u32).collect();
    let b_dims: Vec<u32> = b_shape.dims().iter().map(|&x| x as u32).collect();
    let result_dims: Vec<u32> = result_shape.dims().iter().map(|&x| x as u32).collect();

    let a_strides: Vec<u32> = a_shape.strides().iter().map(|&x| x as u32).collect();
    let b_strides: Vec<u32> = b_shape.strides().iter().map(|&x| x as u32).collect();

    let a_ndim = a_shape.ndim() as u32;
    let b_ndim = b_shape.ndim() as u32;
    let result_ndim = result_shape.ndim() as u32;
    let total_elements_u32 = total_elements as u32;

    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        a_dims.as_ptr() as *mut c_void,
        a_strides.as_ptr() as *mut c_void,
        &a_ndim as *const u32 as *mut c_void,
        b_dims.as_ptr() as *mut c_void,
        b_strides.as_ptr() as *mut c_void,
        &b_ndim as *const u32 as *mut c_void,
        result_dims.as_ptr() as *mut c_void,
        &result_ndim as *const u32 as *mut c_void,
        &total_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn elementwise_div_broadcast<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
) -> Result<()>
where
    T: NumericOps,
{
    elementwise_div_broadcast_async(
        a,
        b,
        result,
        a_shape,
        b_shape,
        result_shape,
        &Stream::new()?,
    )
}

pub fn elementwise_div_broadcast_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    result: &DeviceMemory<T>,
    a_shape: &Shape,
    b_shape: &Shape,
    result_shape: &Shape,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("elementwise_div_broadcast_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let total_elements = result_shape.size();
    let grid_dim = calculate_grid_1d(total_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data for GPU
    let a_dims: Vec<u32> = a_shape.dims().iter().map(|&x| x as u32).collect();
    let b_dims: Vec<u32> = b_shape.dims().iter().map(|&x| x as u32).collect();
    let result_dims: Vec<u32> = result_shape.dims().iter().map(|&x| x as u32).collect();

    let a_strides: Vec<u32> = a_shape.strides().iter().map(|&x| x as u32).collect();
    let b_strides: Vec<u32> = b_shape.strides().iter().map(|&x| x as u32).collect();

    let a_ndim = a_shape.ndim() as u32;
    let b_ndim = b_shape.ndim() as u32;
    let result_ndim = result_shape.ndim() as u32;
    let total_elements_u32 = total_elements as u32;

    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        result.as_ptr() as *mut c_void,
        a_dims.as_ptr() as *mut c_void,
        a_strides.as_ptr() as *mut c_void,
        &a_ndim as *const u32 as *mut c_void,
        b_dims.as_ptr() as *mut c_void,
        b_strides.as_ptr() as *mut c_void,
        &b_ndim as *const u32 as *mut c_void,
        result_dims.as_ptr() as *mut c_void,
        &result_ndim as *const u32 as *mut c_void,
        &total_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Scalar operations
// =============================================================================

pub fn scalar_add<T>(
    input: &DeviceMemory<T>,
    scalar: T,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    scalar_add_async(input, scalar, result, len, &Stream::new()?)
}

pub fn scalar_add_async<T>(
    input: &DeviceMemory<T>,
    scalar: T,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("scalar_add_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &scalar as *const T as *mut c_void,
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn scalar_mul<T>(
    input: &DeviceMemory<T>,
    scalar: T,
    result: &DeviceMemory<T>,
    len: usize,
) -> Result<()>
where
    T: NumericOps,
{
    scalar_mul_async(input, scalar, result, len, &Stream::new()?)
}

pub fn scalar_mul_async<T>(
    input: &DeviceMemory<T>,
    scalar: T,
    result: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("scalar_mul_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &scalar as *const T as *mut c_void,
        result.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Reduction operations
// =============================================================================

pub fn reduce_sum<T>(input: &DeviceMemory<T>, len: usize) -> Result<T>
where
    T: NumericOps,
{
    reduce_sum_async(input, len, &Stream::new()?)
}

pub fn reduce_sum_async<T>(input: &DeviceMemory<T>, len: usize, stream: &Stream) -> Result<T>
where
    T: NumericOps,
{
    let kernel_name = format!("reduce_sum_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);

    let mut temp_result = DeviceMemory::<T>::new(1)?;
    // Initialize result to zero
    temp_result.memset(0)?;

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        temp_result.as_ptr() as *mut c_void,
    ];

    function.launch(
        grid_dim,
        Dim3::new_1d(block_size),
        0,
        Some(stream),
        &mut kernel_args,
    )?;
    stream.synchronize()?;

    let mut result = vec![T::default(); 1];
    temp_result.copy_to_host(&mut result)?;
    Ok(result[0])
}

pub fn reduce_min<T>(input: &DeviceMemory<T>, len: usize) -> Result<T>
where
    T: NumericOps + PartialOrd,
{
    reduce_min_async(input, len, &Stream::new()?)
}

pub fn reduce_min_async<T>(input: &DeviceMemory<T>, len: usize, stream: &Stream) -> Result<T>
where
    T: NumericOps + PartialOrd,
{
    let kernel_name = format!("reduce_min_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);

    let mut temp_result = DeviceMemory::<T>::new(1)?;
    // Initialize with first element
    if len > 0 {
        let first_device = DeviceMemory::<T>::new(1)?;
        temp_result.copy_from_device(&first_device)?;
    }

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        temp_result.as_ptr() as *mut c_void,
    ];

    function.launch(
        grid_dim,
        Dim3::new_1d(block_size),
        0,
        Some(stream),
        &mut kernel_args,
    )?;
    stream.synchronize()?;

    let mut result = vec![T::default(); 1];
    temp_result.copy_to_host(&mut result)?;
    Ok(result[0])
}

// Reduction along specific axis
pub fn reduce_sum_axis<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    axis: usize,
) -> Result<()>
where
    T: NumericOps,
{
    reduce_sum_axis_async(input, output, input_shape, axis, &Stream::new()?)
}

pub fn reduce_sum_axis_async<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    axis: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("reduce_sum_axis_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let output_size = input_shape.size() / input_shape.dims()[axis];
    let grid_dim = calculate_grid_1d(output_size as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data
    let dims: Vec<u32> = input_shape.dims().iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = input_shape.strides().iter().map(|&x| x as u32).collect();
    let ndim = input_shape.ndim() as u32;
    let axis_u32 = axis as u32;
    let axis_size = input_shape.dims()[axis] as u32;

    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        dims.as_ptr() as *mut c_void,
        strides.as_ptr() as *mut c_void,
        &ndim as *const u32 as *mut c_void,
        &axis_u32 as *const u32 as *mut c_void,
        &axis_size as *const u32 as *mut c_void,
        &(output_size as u32) as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Matrix operations
// =============================================================================

pub fn matrix_multiply<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    c: &DeviceMemory<T>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()>
where
    T: NumericOps,
{
    matrix_multiply_async(a, b, c, m, k, n, &Stream::new()?)
}

pub fn matrix_multiply_async<T>(
    a: &DeviceMemory<T>,
    b: &DeviceMemory<T>,
    c: &DeviceMemory<T>,
    m: usize,
    k: usize,
    n: usize,
    stream: &Stream,
) -> Result<()>
where
    T: NumericOps,
{
    let kernel_name = format!("matrix_multiply_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    // Use 2D grid for matrix multiplication
    let block_x = 16;
    let block_y = 16;
    let grid_x = (n as u32 + block_x - 1) / block_x;
    let grid_y = (m as u32 + block_y - 1) / block_y;

    let grid_dim = Dim3::new_2d(grid_x, grid_y);
    let block_dim = Dim3::new_2d(block_x, block_y);

    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let mut kernel_args = [
        a.as_ptr(),
        b.as_ptr(),
        c.as_ptr() as *mut c_void,
        &m_u32 as *const u32 as *mut c_void,
        &k_u32 as *const u32 as *mut c_void,
        &n_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn transpose<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    output_shape: &Shape,
) -> Result<()>
where
    T: TransposableOps,
{
    transpose_async(input, output, input_shape, output_shape, &Stream::new()?)
}

pub fn transpose_async<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    output_shape: &Shape,
    stream: &Stream,
) -> Result<()>
where
    T: TransposableOps,
{
    let kernel_name = format!("transpose_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let total_elements = input_shape.size();
    let grid_dim = calculate_grid_1d(total_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    // Prepare shape data
    let input_dims: Vec<u32> = input_shape.dims().iter().map(|&x| x as u32).collect();
    let output_dims: Vec<u32> = output_shape.dims().iter().map(|&x| x as u32).collect();
    let input_strides: Vec<u32> = input_shape.strides().iter().map(|&x| x as u32).collect();
    let output_strides: Vec<u32> = output_shape.strides().iter().map(|&x| x as u32).collect();

    let ndim = input_shape.ndim() as u32;
    let total_elements_u32 = total_elements as u32;

    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        input_dims.as_ptr() as *mut c_void,
        input_strides.as_ptr() as *mut c_void,
        output_dims.as_ptr() as *mut c_void,
        output_strides.as_ptr() as *mut c_void,
        &ndim as *const u32 as *mut c_void,
        &total_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Indexing and slicing operations
// =============================================================================

pub fn get_element<T>(input: &DeviceMemory<T>, index: usize) -> Result<T>
where
    T: Copy + Default + 'static,
{
    // For single element access, copy to host
    let mut result = vec![T::default(); 1];
    let temp_device = DeviceMemory::<T>::new(1)?;

    // Use copy kernel to get single element
    let kernel_name = "copy_element";
    let function = get_kernel_function(&kernel_name)?;

    let index_u32 = index as u32;
    let mut kernel_args = [
        input.as_ptr(),
        temp_device.as_ptr() as *mut c_void,
        &index_u32 as *const u32 as *mut c_void,
    ];

    function.launch(Dim3::new_1d(1), Dim3::new_1d(1), 0, None, &mut kernel_args)?;
    temp_device.copy_to_host(&mut result)?;
    Ok(result[0])
}

pub fn set_element<T>(output: &mut DeviceMemory<T>, index: usize, value: T) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let kernel_name = "set_element";
    let function = get_kernel_function(&kernel_name)?;

    let index_u32 = index as u32;
    let mut kernel_args = [
        output.as_ptr() as *mut c_void,
        &index_u32 as *const u32 as *mut c_void,
        &value as *const T as *mut c_void,
    ];

    function.launch(Dim3::new_1d(1), Dim3::new_1d(1), 0, None, &mut kernel_args)?;
    Ok(())
}

pub fn slice_first_dim<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    start: usize,
    end: usize,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    slice_first_dim_async(input, output, input_shape, start, end, &Stream::new()?)
}

pub fn slice_first_dim_async<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    start: usize,
    end: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let kernel_name = "slice_first_dim";
    let function = get_kernel_function(&kernel_name)?;

    let slice_len = end - start;
    let elements_per_slice = input_shape.size() / input_shape.dims()[0];
    let total_output_elements = slice_len * elements_per_slice;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(total_output_elements as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let start_u32 = start as u32;
    let slice_len_u32 = slice_len as u32;
    let elements_per_slice_u32 = elements_per_slice as u32;
    let total_output_elements_u32 = total_output_elements as u32;

    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        &start_u32 as *const u32 as *mut c_void,
        &slice_len_u32 as *const u32 as *mut c_void,
        &elements_per_slice_u32 as *const u32 as *mut c_void,
        &total_output_elements_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn extract_column<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    col_index: usize,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    extract_column_async(input, output, input_shape, col_index, &Stream::new()?)
}

pub fn extract_column_async<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    input_shape: &Shape,
    col_index: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let kernel_name = "extract_column";
    let function = get_kernel_function(&kernel_name)?;

    if input_shape.ndim() != 2 {
        return Err(crate::error::Error::InvalidOperation(
            "Extract column requires 2D array".to_string(),
        ));
    }

    let rows = input_shape.dims()[0];
    let cols = input_shape.dims()[1];

    let block_size = 256;
    let grid_dim = calculate_grid_1d(rows as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let col_index_u32 = col_index as u32;

    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        &rows_u32 as *const u32 as *mut c_void,
        &cols_u32 as *const u32 as *mut c_void,
        &col_index_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Map, filter, reduce operations
// =============================================================================

pub fn map<T, U, F>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<U>,
    len: usize,
    _func: F,
) -> Result<()>
where
    T: Mappable<U>,
    U: Copy + Default + 'static,
    F: Fn(T) -> U,
{
    // In a real implementation, you'd need to compile the function into a kernel
    // For now, this is a placeholder that would require kernel generation
    let function = get_kernel_function(T::map_kernel_name())?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, None, &mut kernel_args)?;
    Ok(())
}

pub fn filter<T, F>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    len: usize,
    _predicate: F,
) -> Result<usize>
where
    T: Filterable,
    F: Fn(T) -> bool,
{
    // In a real implementation, you'd need stream compaction algorithms
    // This is a simplified placeholder
    let function = get_kernel_function(T::filter_kernel_name())?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let mut count_buffer = DeviceMemory::<u32>::new(1)?;
    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        output.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
        count_buffer.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, None, &mut kernel_args)?;

    let mut count = vec![0u32; 1];
    count_buffer.copy_to_host(&mut count)?;
    Ok(count[0] as usize)
}

pub fn reduce<T, F>(input: &DeviceMemory<T>, len: usize, initial: T, _func: F) -> Result<T>
where
    T: Reducible,
    F: Fn(T, T) -> T,
{
    // Similar to sum, but with custom operation
    let function = get_kernel_function(T::reduce_kernel_name())?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);

    let mut temp_result = DeviceMemory::<T>::new(1)?;
    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        &initial as *const T as *mut c_void,
        temp_result.as_ptr() as *mut c_void,
    ];

    function.launch(
        grid_dim,
        Dim3::new_1d(block_size),
        0,
        None,
        &mut kernel_args,
    )?;

    let mut result = vec![T::default(); 1];
    temp_result.copy_to_host(&mut result)?;
    Ok(result[0])
}

pub fn find_index<T, F>(input: &DeviceMemory<T>, len: usize, _predicate: F) -> Result<Option<usize>>
where
    T: Searchable,
    F: Fn(T) -> bool,
{
    let function = get_kernel_function(T::search_kernel_name())?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);

    let mut index_buffer = DeviceMemory::<i32>::new(1)?;
    // Initialize to -1 (not found)
    let not_found = -1i32;
    index_buffer.copy_from_host(&[not_found])?;

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        index_buffer.as_ptr() as *mut c_void,
    ];

    function.launch(
        grid_dim,
        Dim3::new_1d(block_size),
        0,
        None,
        &mut kernel_args,
    )?;

    let mut index = vec![-1i32; 1];
    index_buffer.copy_to_host(&mut index)?;

    if index[0] >= 0 {
        Ok(Some(index[0] as usize))
    } else {
        Ok(None)
    }
}

// =============================================================================
// Range operations
// =============================================================================

pub fn calculate_range_len<T>(start: T, end: T, step: T) -> Result<usize>
where
    T: RangeOps + PartialOrd + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + Into<f64>,
{
    if step.into() == 0.0 {
        return Err(crate::error::Error::InvalidOperation(
            "Step cannot be zero".to_string(),
        ));
    }

    let diff = end - start;
    let len = diff / step;
    Ok(len.into().ceil() as usize)
}

pub fn fill_range<T>(output: &DeviceMemory<T>, start: T, step: T, len: usize) -> Result<()>
where
    T: RangeOps,
{
    fill_range_async(output, start, step, len, &Stream::new()?)
}

pub fn fill_range_async<T>(
    output: &DeviceMemory<T>,
    start: T,
    step: T,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: RangeOps,
{
    let function = get_kernel_function(T::range_kernel_name())?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        &start as *const T as *mut c_void,
        &step as *const T as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
        output.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn fill_linspace(output: &DeviceMemory<f64>, start: f64, step: f64, len: usize) -> Result<()> {
    fill_linspace_async(output, start, step, len, &Stream::new()?)
}

pub fn fill_linspace_async(
    output: &DeviceMemory<f64>,
    start: f64,
    step: f64,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    let function = get_kernel_function("linspace_double")?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        &start as *const f64 as *mut c_void,
        &step as *const f64 as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
        output.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

// =============================================================================
// Utility functions
// =============================================================================

pub fn copy_memory<T>(src: &DeviceMemory<T>, dst: &DeviceMemory<T>, len: usize) -> Result<()>
where
    T: Copy + Default + 'static,
{
    copy_memory_async(src, dst, len, &Stream::new()?)
}

pub fn copy_memory_async<T>(
    src: &DeviceMemory<T>,
    dst: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let function = get_kernel_function("copy_memory")?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        src.as_ptr(),
        dst.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn fill_value<T>(output: &DeviceMemory<T>, value: T, len: usize) -> Result<()>
where
    T: Copy + Default + 'static,
{
    fill_value_async(output, value, len, &Stream::new()?)
}

pub fn fill_value_async<T>(
    output: &DeviceMemory<T>,
    value: T,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let function = get_kernel_function("fill_value")?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        output.as_ptr() as *mut c_void,
        &value as *const T as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args)?;
    Ok(())
}

pub fn reduce_max<T>(input: &DeviceMemory<T>, len: usize) -> Result<T>
where
    T: NumericOps + PartialOrd,
{
    reduce_max_async(input, len, &Stream::new()?)
}

pub fn reduce_max_async<T>(input: &DeviceMemory<T>, len: usize, stream: &Stream) -> Result<T>
where
    T: NumericOps + PartialOrd,
{
    let kernel_name = format!("reduce_max_{}", T::TYPE_NAME);
    let function = get_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);

    let mut temp_result = DeviceMemory::<T>::new(1)?;
    // Initialize with first element
    if len > 0 {
        let mut first_element = vec![T::default(); 1];
        let first_device = DeviceMemory::<T>::new(1)?;
        // Copy first element to initialize result
        temp_result.copy_from_device(&first_device)?;
    }

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        temp_result.as_ptr() as *mut c_void,
    ];

    function.launch(
        grid_dim,
        Dim3::new_1d(block_size),
        0,
        Some(stream),
        &mut kernel_args,
    )?;
    stream.synchronize()?;

    let mut result = vec![T::default(); 1];
    temp_result.copy_to_host(&mut result)?;
    Ok(result[0])
}
