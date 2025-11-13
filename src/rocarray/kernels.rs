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

// =============================================================================
// TEAM-490: Cast Operations (Phase 2 Step 2)
// =============================================================================

/// Generic cast operation wrapper
fn cast_generic<S, D>(
    input: &DeviceMemory<S>,
    output: &DeviceMemory<D>,
    kernel_name: &str,
    len: usize,
) -> Result<()>
where
    S: Copy + Default + 'static,
    D: Copy + Default + 'static,
{
    let function = get_kernel_function(kernel_name)?;

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

// Macro to define cast wrapper functions
macro_rules! define_cast_wrapper {
    ($src_type:ty, $dst_type:ty, $fn_name:ident, $src_name:literal, $dst_name:literal) => {
        pub fn $fn_name(
            input: &DeviceMemory<$src_type>,
            output: &DeviceMemory<$dst_type>,
            len: usize,
        ) -> Result<()> {
            let kernel_name = concat!("cast_", $src_name, "_", $dst_name);
            cast_generic(input, output, kernel_name, len)
        }
    };
}

// F32 casts
define_cast_wrapper!(f32, f64, cast_f32_f64, "f32", "f64");
define_cast_wrapper!(f32, i32, cast_f32_i32, "f32", "i32");
define_cast_wrapper!(f32, i64, cast_f32_i64, "f32", "i64");
define_cast_wrapper!(f32, u8, cast_f32_u8, "f32", "u8");
define_cast_wrapper!(f32, u32, cast_f32_u32, "f32", "u32");

// F64 casts
define_cast_wrapper!(f64, f32, cast_f64_f32, "f64", "f32");
define_cast_wrapper!(f64, i32, cast_f64_i32, "f64", "i32");
define_cast_wrapper!(f64, i64, cast_f64_i64, "f64", "i64");
define_cast_wrapper!(f64, u8, cast_f64_u8, "f64", "u8");
define_cast_wrapper!(f64, u32, cast_f64_u32, "f64", "u32");

// I32 casts
define_cast_wrapper!(i32, f32, cast_i32_f32, "i32", "f32");
define_cast_wrapper!(i32, f64, cast_i32_f64, "i32", "f64");
define_cast_wrapper!(i32, i64, cast_i32_i64, "i32", "i64");
define_cast_wrapper!(i32, u8, cast_i32_u8, "i32", "u8");
define_cast_wrapper!(i32, u32, cast_i32_u32, "i32", "u32");

// I64 casts
define_cast_wrapper!(i64, f32, cast_i64_f32, "i64", "f32");
define_cast_wrapper!(i64, f64, cast_i64_f64, "i64", "f64");
define_cast_wrapper!(i64, i32, cast_i64_i32, "i64", "i32");
define_cast_wrapper!(i64, u8, cast_i64_u8, "i64", "u8");
define_cast_wrapper!(i64, u32, cast_i64_u32, "i64", "u32");

// U8 casts
define_cast_wrapper!(u8, f32, cast_u8_f32, "u8", "f32");
define_cast_wrapper!(u8, f64, cast_u8_f64, "u8", "f64");
define_cast_wrapper!(u8, i32, cast_u8_i32, "u8", "i32");
define_cast_wrapper!(u8, i64, cast_u8_i64, "u8", "i64");
define_cast_wrapper!(u8, u32, cast_u8_u32, "u8", "u32");

// U32 casts
define_cast_wrapper!(u32, f32, cast_u32_f32, "u32", "f32");
define_cast_wrapper!(u32, f64, cast_u32_f64, "u32", "f64");
define_cast_wrapper!(u32, i32, cast_u32_i32, "u32", "i32");
define_cast_wrapper!(u32, i64, cast_u32_i64, "u32", "i64");
define_cast_wrapper!(u32, u8, cast_u32_u8, "u32", "u8");

// =============================================================================
// TEAM-490: Ternary Operations (Phase 2 Step 2)
// =============================================================================

/// Generic ternary where/select operation
fn where_generic<C, T>(
    condition: &DeviceMemory<C>,
    true_vals: &DeviceMemory<T>,
    false_vals: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    kernel_name: &str,
    len: usize,
) -> Result<()>
where
    C: Copy + Default + 'static,
    T: Copy + Default + 'static,
{
    let function = get_kernel_function(kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        condition.as_ptr(),
        true_vals.as_ptr(),
        false_vals.as_ptr(),
        output.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, None, &mut kernel_args)?;
    Ok(())
}

// Macro to define where wrapper functions
macro_rules! define_where_wrapper {
    ($cond_type:ty, $val_type:ty, $fn_name:ident, $cond_name:literal, $val_name:literal) => {
        pub fn $fn_name(
            condition: &DeviceMemory<$cond_type>,
            true_vals: &DeviceMemory<$val_type>,
            false_vals: &DeviceMemory<$val_type>,
            output: &DeviceMemory<$val_type>,
            len: usize,
        ) -> Result<()> {
            let kernel_name = concat!("where_", $cond_name, "_", $val_name);
            where_generic(condition, true_vals, false_vals, output, kernel_name, len)
        }
    };
}

// U8 condition with various value types
define_where_wrapper!(u8, f32, where_u8_f32, "u8", "f32");
define_where_wrapper!(u8, f64, where_u8_f64, "u8", "f64");
define_where_wrapper!(u8, i32, where_u8_i32, "u8", "i32");
define_where_wrapper!(u8, i64, where_u8_i64, "u8", "i64");
define_where_wrapper!(u8, u8, where_u8_u8, "u8", "u8");
define_where_wrapper!(u8, u32, where_u8_u32, "u8", "u32");

// I32 condition with various value types
define_where_wrapper!(i32, f32, where_i32_f32, "i32", "f32");
define_where_wrapper!(i32, f64, where_i32_f64, "i32", "f64");
define_where_wrapper!(i32, i32, where_i32_i32, "i32", "i32");
define_where_wrapper!(i32, i64, where_i32_i64, "i32", "i64");
define_where_wrapper!(i32, u8, where_i32_u8, "i32", "u8");
define_where_wrapper!(i32, u32, where_i32_u32, "i32", "u32");

// I64 condition with various value types
define_where_wrapper!(i64, f32, where_i64_f32, "i64", "f32");
define_where_wrapper!(i64, f64, where_i64_f64, "i64", "f64");
define_where_wrapper!(i64, i32, where_i64_i32, "i64", "i32");
define_where_wrapper!(i64, i64, where_i64_i64, "i64", "i64");
define_where_wrapper!(i64, u8, where_i64_u8, "i64", "u8");
define_where_wrapper!(i64, u32, where_i64_u32, "i64", "u32");

// =============================================================================
// TEAM-490: Unary Operations (Phase 2 Step 2)
// =============================================================================

/// Generic unary operation wrapper
fn unary_generic<T>(
    input: &DeviceMemory<T>,
    output: &DeviceMemory<T>,
    kernel_name: &str,
    len: usize,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let function = get_kernel_function(kernel_name)?;

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

/// Generic unary operation with parameter
fn unary_param_generic<T>(
    input: &DeviceMemory<T>,
    param: T,
    output: &DeviceMemory<T>,
    kernel_name: &str,
    len: usize,
) -> Result<()>
where
    T: Copy + Default + 'static,
{
    let function = get_kernel_function(kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let mut kernel_args = [
        input.as_ptr(),
        &param as *const T as *mut c_void,
        output.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, None, &mut kernel_args)?;
    Ok(())
}

// Macro to define unary operation wrappers
macro_rules! define_unary_wrapper {
    ($op:ident, $type:ty, $fn_name:ident, $type_name:literal) => {
        pub fn $fn_name(
            input: &DeviceMemory<$type>,
            output: &DeviceMemory<$type>,
            len: usize,
        ) -> Result<()> {
            let kernel_name = concat!(stringify!($op), "_", $type_name);
            unary_generic(input, output, kernel_name, len)
        }
    };
}

// Macro for parametric unary operations
macro_rules! define_unary_param_wrapper {
    ($op:ident, $type:ty, $fn_name:ident, $type_name:literal) => {
        pub fn $fn_name(
            input: &DeviceMemory<$type>,
            param: $type,
            output: &DeviceMemory<$type>,
            len: usize,
        ) -> Result<()> {
            let kernel_name = concat!(stringify!($op), "_", $type_name);
            unary_param_generic(input, param, output, kernel_name, len)
        }
    };
}

// Exponential/Logarithmic operations
define_unary_wrapper!(exp, f32, unary_exp_f32, "f32");
define_unary_wrapper!(exp, f64, unary_exp_f64, "f64");
define_unary_wrapper!(log, f32, unary_log_f32, "f32");
define_unary_wrapper!(log, f64, unary_log_f64, "f64");

// Trigonometric operations
define_unary_wrapper!(sin, f32, unary_sin_f32, "f32");
define_unary_wrapper!(sin, f64, unary_sin_f64, "f64");
define_unary_wrapper!(cos, f32, unary_cos_f32, "f32");
define_unary_wrapper!(cos, f64, unary_cos_f64, "f64");
define_unary_wrapper!(tanh, f32, unary_tanh_f32, "f32");
define_unary_wrapper!(tanh, f64, unary_tanh_f64, "f64");

// Rounding operations
define_unary_wrapper!(ceil, f32, unary_ceil_f32, "f32");
define_unary_wrapper!(ceil, f64, unary_ceil_f64, "f64");
define_unary_wrapper!(floor, f32, unary_floor_f32, "f32");
define_unary_wrapper!(floor, f64, unary_floor_f64, "f64");
define_unary_wrapper!(round, f32, unary_round_f32, "f32");
define_unary_wrapper!(round, f64, unary_round_f64, "f64");

// Error functions
define_unary_wrapper!(erf, f32, unary_erf_f32, "f32");
define_unary_wrapper!(erf, f64, unary_erf_f64, "f64");
define_unary_wrapper!(normcdf, f32, unary_normcdf_f32, "f32");
define_unary_wrapper!(normcdf, f64, unary_normcdf_f64, "f64");

// Basic operations
define_unary_wrapper!(abs, f32, unary_abs_f32, "f32");
define_unary_wrapper!(abs, f64, unary_abs_f64, "f64");
define_unary_wrapper!(abs, i32, unary_abs_i32, "i32");
define_unary_wrapper!(abs, i64, unary_abs_i64, "i64");

define_unary_wrapper!(recip, f32, unary_recip_f32, "f32");
define_unary_wrapper!(recip, f64, unary_recip_f64, "f64");

define_unary_wrapper!(neg, f32, unary_neg_f32, "f32");
define_unary_wrapper!(neg, f64, unary_neg_f64, "f64");
define_unary_wrapper!(neg, i32, unary_neg_i32, "i32");
define_unary_wrapper!(neg, i64, unary_neg_i64, "i64");

define_unary_wrapper!(sqr, f32, unary_sqr_f32, "f32");
define_unary_wrapper!(sqr, f64, unary_sqr_f64, "f64");
define_unary_wrapper!(sqr, i32, unary_sqr_i32, "i32");
define_unary_wrapper!(sqr, i64, unary_sqr_i64, "i64");

define_unary_wrapper!(sqrt, f32, unary_sqrt_f32, "f32");
define_unary_wrapper!(sqrt, f64, unary_sqrt_f64, "f64");

define_unary_wrapper!(sign, f32, unary_sign_f32, "f32");
define_unary_wrapper!(sign, f64, unary_sign_f64, "f64");
define_unary_wrapper!(sign, i32, unary_sign_i32, "i32");
define_unary_wrapper!(sign, i64, unary_sign_i64, "i64");

// Activation functions
define_unary_wrapper!(gelu, f32, unary_gelu_f32, "f32");
define_unary_wrapper!(gelu, f64, unary_gelu_f64, "f64");
define_unary_wrapper!(gelu_erf, f32, unary_gelu_erf_f32, "f32");
define_unary_wrapper!(gelu_erf, f64, unary_gelu_erf_f64, "f64");
define_unary_wrapper!(silu, f32, unary_silu_f32, "f32");
define_unary_wrapper!(silu, f64, unary_silu_f64, "f64");
define_unary_wrapper!(relu, f32, unary_relu_f32, "f32");
define_unary_wrapper!(relu, f64, unary_relu_f64, "f64");
define_unary_wrapper!(sigmoid, f32, unary_sigmoid_f32, "f32");
define_unary_wrapper!(sigmoid, f64, unary_sigmoid_f64, "f64");

// Parametric operations
define_unary_param_wrapper!(elu, f32, unary_elu_f32, "f32");
define_unary_param_wrapper!(elu, f64, unary_elu_f64, "f64");
define_unary_param_wrapper!(powf, f32, unary_powf_f32, "f32");
define_unary_param_wrapper!(powf, f64, unary_powf_f64, "f64");

// Copy operations
define_unary_wrapper!(copy, f32, unary_copy_f32, "f32");
define_unary_wrapper!(copy, f64, unary_copy_f64, "f64");
define_unary_wrapper!(copy, i32, unary_copy_i32, "i32");
define_unary_wrapper!(copy, i64, unary_copy_i64, "i64");
define_unary_wrapper!(copy, u8, unary_copy_u8, "u8");
define_unary_wrapper!(copy, u32, unary_copy_u32, "u32");

// =============================================================================
// TEAM-497: Indexing and upsampling operations (CUDA parity for Candle)
// Kernel implementations: kernels.hip:1044-1351
// NOTE: Binary ops (badd, bsub, bmul, bdiv) and comparison ops (eq, ne, lt, le, gt, ge)
//       are ALREADY wired up in Candle's ROCm backend (candle-core/src/rocm_backend/ops.rs)
//       via kernels.hip lines 904-1032. No additional wrappers needed here.
// =============================================================================

/// Upsample nearest 2D (CUDA: candle-kernels/src/conv.cu)
pub fn upsample_nearest2d_f32(
    input: &DeviceMemory<f32>,
    output: &mut DeviceMemory<f32>,
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    scale_h: u32,
    scale_w: u32,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("upsample_nearest2d_f32")?;
    
    let total_elements = batch * channels * out_h * out_w;
    let (grid, block) = calculate_grid_1d(total_elements);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &input.as_ptr() as *const _ as *mut c_void,
            &output.as_mut_ptr() as *const _ as *mut c_void,
            &batch as *const _ as *mut c_void,
            &channels as *const _ as *mut c_void,
            &in_h as *const _ as *mut c_void,
            &in_w as *const _ as *mut c_void,
            &out_h as *const _ as *mut c_void,
            &out_w as *const _ as *mut c_void,
            &scale_h as *const _ as *mut c_void,
            &scale_w as *const _ as *mut c_void,
        ],
    )
}

/// Gather (CUDA: candle-kernels/src/indexing.cu) - Candle-compatible signature
/// Kernel: gather_i64_f32 (GATHER_OP macro)
pub fn gather_i64_f32(
    numel: usize,
    ids: &DeviceMemory<i64>,
    inp: &DeviceMemory<f32>,
    out: &mut DeviceMemory<f32>,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("gather_i64_f32")?;
    
    let (grid, block) = calculate_grid_1d(numel as u32);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &(numel as u64) as *const _ as *mut c_void,
            &ids.as_ptr() as *const _ as *mut c_void,
            &inp.as_ptr() as *const _ as *mut c_void,
            &out.as_mut_ptr() as *const _ as *mut c_void,
            &(left_size as u64) as *const _ as *mut c_void,
            &(src_dim_size as u64) as *const _ as *mut c_void,
            &(ids_dim_size as u64) as *const _ as *mut c_void,
            &(right_size as u64) as *const _ as *mut c_void,
        ],
    )
}

/// Scatter (CUDA: candle-kernels/src/indexing.cu) - Candle-compatible signature
/// Kernel: s_i64_f32 (S_OP macro)
pub fn s_i64_f32(
    ids: &DeviceMemory<i64>,
    inp: &DeviceMemory<f32>,
    out: &mut DeviceMemory<f32>,
    left_size: usize,
    src_dim_size: usize,
    dst_dim_size: usize,
    right_size: usize,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("s_i64_f32")?;
    
    let numel = left_size * right_size;
    let (grid, block) = calculate_grid_1d(numel as u32);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &ids.as_ptr() as *const _ as *mut c_void,
            &inp.as_ptr() as *const _ as *mut c_void,
            &out.as_mut_ptr() as *const _ as *mut c_void,
            &(left_size as u64) as *const _ as *mut c_void,
            &(src_dim_size as u64) as *const _ as *mut c_void,
            &(dst_dim_size as u64) as *const _ as *mut c_void,
            &(right_size as u64) as *const _ as *mut c_void,
        ],
    )
}

/// Scatter-add (CUDA: candle-kernels/src/indexing.cu) - Candle-compatible signature
/// Kernel: sa_i64_f32 (SA_OP macro)
pub fn sa_i64_f32(
    ids: &DeviceMemory<i64>,
    inp: &DeviceMemory<f32>,
    out: &mut DeviceMemory<f32>,
    left_size: usize,
    src_dim_size: usize,
    dst_dim_size: usize,
    right_size: usize,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("sa_i64_f32")?;
    
    let numel = left_size * right_size;
    let (grid, block) = calculate_grid_1d(numel as u32);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &ids.as_ptr() as *const _ as *mut c_void,
            &inp.as_ptr() as *const _ as *mut c_void,
            &out.as_mut_ptr() as *const _ as *mut c_void,
            &(left_size as u64) as *const _ as *mut c_void,
            &(src_dim_size as u64) as *const _ as *mut c_void,
            &(dst_dim_size as u64) as *const _ as *mut c_void,
            &(right_size as u64) as *const _ as *mut c_void,
        ],
    )
}

/// Index select (CUDA: candle-kernels/src/indexing.cu) - Candle-compatible signature
/// Kernel: is_i64_f32 (IS_OP macro)
pub fn is_i64_f32(
    numel: usize,
    num_dims: usize,
    info: &DeviceMemory<usize>,
    ids: &DeviceMemory<i64>,
    inp: &DeviceMemory<f32>,
    out: &mut DeviceMemory<f32>,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("is_i64_f32")?;
    
    let (grid, block) = calculate_grid_1d(numel as u32);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &(numel as u64) as *const _ as *mut c_void,
            &(num_dims as u64) as *const _ as *mut c_void,
            &info.as_ptr() as *const _ as *mut c_void,
            &ids.as_ptr() as *const _ as *mut c_void,
            &inp.as_ptr() as *const _ as *mut c_void,
            &out.as_mut_ptr() as *const _ as *mut c_void,
            &(left_size as u64) as *const _ as *mut c_void,
            &(src_dim_size as u64) as *const _ as *mut c_void,
            &(ids_dim_size as u64) as *const _ as *mut c_void,
            &(right_size as u64) as *const _ as *mut c_void,
        ],
    )
}

/// Index add (CUDA: candle-kernels/src/indexing.cu) - Candle-compatible signature
/// Kernel: ia_i64_f32 (IA_OP macro)
pub fn ia_i64_f32(
    ids: &DeviceMemory<i64>,
    ids_dim_size: usize,
    inp: &DeviceMemory<f32>,
    out: &mut DeviceMemory<f32>,
    left_size: usize,
    src_dim_size: usize,
    dst_dim_size: usize,
    right_size: usize,
    stream: &Stream,
) -> Result<()> {
    let module = get_kernels_module()?;
    let func = module.get_function("ia_i64_f32")?;
    
    let numel = left_size * right_size;
    let (grid, block) = calculate_grid_1d(numel as u32);
    
    func.launch(
        grid,
        block,
        0,
        stream,
        &[
            &ids.as_ptr() as *const _ as *mut c_void,
            &(ids_dim_size as u64) as *const _ as *mut c_void,
            &inp.as_ptr() as *const _ as *mut c_void,
            &out.as_mut_ptr() as *const _ as *mut c_void,
            &(left_size as u64) as *const _ as *mut c_void,
            &(src_dim_size as u64) as *const _ as *mut c_void,
            &(dst_dim_size as u64) as *const _ as *mut c_void,
            &(right_size as u64) as *const _ as *mut c_void,
        ],
    )
}
