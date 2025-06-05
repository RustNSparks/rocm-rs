// src/rocarray/sorting.rs
use crate::error::Result;
use crate::hip::{DeviceMemory, Stream, Module, Function, calculate_grid_1d, Dim3};
use std::ffi::c_void;
use std::sync::Once;

static INIT_SORT: Once = Once::new();
static mut SORT_MODULE: Option<Module> = None;

/// Trait for types that can be sorted on GPU
pub trait Sortable: Copy + Default + PartialOrd + 'static {
    const TYPE_NAME: &'static str;
    const SIZE: usize = std::mem::size_of::<Self>();
}

// Implement Sortable for common types
impl Sortable for f32 {
    const TYPE_NAME: &'static str = "float";
}

impl Sortable for f64 {
    const TYPE_NAME: &'static str = "double";
}

impl Sortable for i32 {
    const TYPE_NAME: &'static str = "int";
}

impl Sortable for u32 {
    const TYPE_NAME: &'static str = "uint";
}

impl Sortable for i64 {
    const TYPE_NAME: &'static str = "long";
}

impl Sortable for u64 {
    const TYPE_NAME: &'static str = "ulong";
}

impl Sortable for i16 {
    const TYPE_NAME: &'static str = "short";
}

impl Sortable for u16 {
    const TYPE_NAME: &'static str = "ushort";
}

impl Sortable for i8 {
    const TYPE_NAME: &'static str = "char";
}

impl Sortable for u8 {
    const TYPE_NAME: &'static str = "uchar";
}

// Initialize sorting kernels
fn init_sort_kernels() -> Result<()> {
    INIT_SORT.call_once(|| {
        let kernel_source = include_str!("sorting_kernels.hip");

        match crate::hip::compile_and_load(kernel_source, &[]) {
            Ok(module) => unsafe {
                SORT_MODULE = Some(module);
            },
            Err(e) => {
                eprintln!("Failed to load sorting kernels: {:?}", e);
            }
        }
    });
    Ok(())
}

fn get_sort_kernel_function(name: &str) -> Result<Function> {
    init_sort_kernels()?;

    unsafe {
        if let Some(ref module) = SORT_MODULE {
            module.get_function(name)
        } else {
            Err(crate::error::Error::InvalidOperation(
                "Sort kernels not initialized".to_string()
            ))
        }
    }
}

/// Sort array in ascending order using GPU-based radix sort
pub fn sort_ascending<T>(data: &mut DeviceMemory<T>, len: usize) -> Result<()>
where
    T: Sortable,
{
    sort_ascending_async(data, len, &Stream::new()?)
}

/// Sort array in ascending order asynchronously
pub fn sort_ascending_async<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    if len <= 1 {
        return Ok(());
    }

    // For small arrays, use bitonic sort
    if len <= 1024 {
        bitonic_sort_ascending(data, len, stream)
    } else {
        // For larger arrays, use radix sort
        radix_sort_ascending(data, len, stream)
    }
}

/// Sort array in descending order
pub fn sort_descending<T>(data: &mut DeviceMemory<T>, len: usize) -> Result<()>
where
    T: Sortable,
{
    sort_descending_async(data, len, &Stream::new()?)
}

/// Sort array in descending order asynchronously
pub fn sort_descending_async<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    if len <= 1 {
        return Ok(());
    }

    // For small arrays, use bitonic sort
    if len <= 1024 {
        bitonic_sort_descending(data, len, stream)
    } else {
        // For larger arrays, use radix sort followed by reverse
        radix_sort_ascending(data, len, stream)?;
        reverse_array(data, len, stream)
    }
}

/// Bitonic sort for small arrays (up to 1024 elements)
fn bitonic_sort_ascending<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("bitonic_sort_ascending_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    // Bitonic sort works on power-of-2 sizes, so pad if necessary
    let padded_len = len.next_power_of_two();

    let block_size = 256.min(padded_len);
    let grid_dim = calculate_grid_1d(padded_len as u32, block_size as u32);
    let block_dim = Dim3::new_1d(block_size as u32);

    let len_u32 = len as u32;
    let padded_len_u32 = padded_len as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        &padded_len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Bitonic sort for small arrays in descending order
fn bitonic_sort_descending<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("bitonic_sort_descending_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let padded_len = len.next_power_of_two();

    let block_size = 256.min(padded_len);
    let grid_dim = calculate_grid_1d(padded_len as u32, block_size as u32);
    let block_dim = Dim3::new_1d(block_size as u32);

    let len_u32 = len as u32;
    let padded_len_u32 = padded_len as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        &padded_len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Radix sort for larger arrays
fn radix_sort_ascending<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("radix_sort_ascending_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    // Allocate temporary buffer for radix sort
    let temp_buffer = DeviceMemory::<T>::new(len)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let kernel_args = [
        data.as_ptr(),
        temp_buffer.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Reverse array in-place
fn reverse_array<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("reverse_array_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d((len / 2) as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Get sorted indices (argsort)
pub fn argsort<T>(
    data: &DeviceMemory<T>,
    indices: &DeviceMemory<u32>,
    len: usize,
) -> Result<()>
where
    T: Sortable,
{
    argsort_async(data, indices, len, &Stream::new()?)
}

/// Get sorted indices asynchronously
pub fn argsort_async<T>(
    data: &DeviceMemory<T>,
    indices: &DeviceMemory<u32>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("argsort_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    // First, initialize indices to 0, 1, 2, ...
    let init_kernel = get_sort_kernel_function("init_indices")?;
    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let init_args = [
        indices.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    init_kernel.launch(grid_dim, block_dim, 0, Some(stream), &mut init_args.clone())?;

    // Then sort indices based on data values
    let sort_args = [
        data.as_ptr(),
        indices.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut sort_args.clone())?;
    Ok(())
}

/// Check if array is sorted in ascending order
pub fn is_sorted<T>(data: &DeviceMemory<T>, len: usize) -> Result<bool>
where
    T: Sortable,
{
    is_sorted_async(data, len, &Stream::new()?)
}

/// Check if array is sorted asynchronously
pub fn is_sorted_async<T>(
    data: &DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<bool>
where
    T: Sortable,
{
    if len <= 1 {
        return Ok(true);
    }

    let kernel_name = format!("is_sorted_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let mut result_buffer = DeviceMemory::<u32>::new(1)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        result_buffer.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;

    stream.synchronize()?;

    let mut result = vec![0u32; 1];
    result_buffer.copy_to_host(&mut result)?;

    Ok(result[0] != 0)
}

/// Partial sort (sort only the first k elements)
pub fn partial_sort<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    k: usize,
) -> Result<()>
where
    T: Sortable,
{
    partial_sort_async(data, len, k, &Stream::new()?)
}

/// Partial sort asynchronously
pub fn partial_sort_async<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    k: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    if k >= len {
        return sort_ascending_async(data, len, stream);
    }

    let kernel_name = format!("partial_sort_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let k_u32 = k as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        &k_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Find the k-th smallest element (quickselect algorithm)
pub fn nth_element<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    n: usize,
) -> Result<T>
where
    T: Sortable,
{
    nth_element_async(data, len, n, &Stream::new()?)
}

/// Find the k-th smallest element asynchronously
pub fn nth_element_async<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    n: usize,
    stream: &Stream,
) -> Result<T>
where
    T: Sortable,
{
    if n >= len {
        return Err(InvalidOperation(
            "Index n is out of bounds".to_string()
        ));
    }

    let kernel_name = format!("nth_element_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let mut result_buffer = DeviceMemory::<T>::new(1)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let n_u32 = n as u32;
    let kernel_args = [
        data.as_ptr(),
        &len_u32 as *const u32 as *mut c_void,
        &n_u32 as *const u32 as *mut c_void,
        result_buffer.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;

    stream.synchronize()?;

    let mut result = vec![T::default(); 1];
    result_buffer.copy_to_host(&mut result)?;

    Ok(result[0])
}

/// Merge two sorted arrays
pub fn merge_sorted<T>(
    left: &DeviceMemory<T>,
    left_len: usize,
    right: &DeviceMemory<T>,
    right_len: usize,
    output: &DeviceMemory<T>,
) -> Result<()>
where
    T: Sortable,
{
    merge_sorted_async(left, left_len, right, right_len, output, &Stream::new()?)
}

/// Merge two sorted arrays asynchronously
pub fn merge_sorted_async<T>(
    left: &DeviceMemory<T>,
    left_len: usize,
    right: &DeviceMemory<T>,
    right_len: usize,
    output: &DeviceMemory<T>,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    let kernel_name = format!("merge_sorted_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let total_len = left_len + right_len;
    let block_size = 256;
    let grid_dim = calculate_grid_1d(total_len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let left_len_u32 = left_len as u32;
    let right_len_u32 = right_len as u32;
    let kernel_args = [
        left.as_ptr(),
        &left_len_u32 as *const u32 as *mut c_void,
        right.as_ptr(),
        &right_len_u32 as *const u32 as *mut c_void,
        output.as_ptr() as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

/// Stable sort (maintains relative order of equal elements)
pub fn stable_sort<T>(data: &mut DeviceMemory<T>, len: usize) -> Result<()>
where
    T: Sortable,
{
    stable_sort_async(data, len, &Stream::new()?)
}

/// Stable sort asynchronously
pub fn stable_sort_async<T>(
    data: &mut DeviceMemory<T>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    // Use merge sort for stability
    let kernel_name = format!("stable_sort_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let temp_buffer = DeviceMemory::<T>::new(len)?;

    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let kernel_args = [
        data.as_ptr(),
        temp_buffer.as_ptr() as *mut c_void,
        &len_u32 as *const u32 as *mut c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut kernel_args.clone())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocarray::ROCArray;

    #[test]
    fn test_sort_ascending() -> Result<()> {
        let data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        let mut arr = ROCArray::from_vec(data)?;

        arr.sort()?;

        let result = arr.to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        Ok(())
    }

    #[test]
    fn test_sort_descending() -> Result<()> {
        let data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        let mut arr = ROCArray::from_vec(data)?;

        arr.sort_descending()?;

        let result = arr.to_vec()?;
        assert_eq!(result, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);

        Ok(())
    }

    #[test]
    fn test_is_sorted() -> Result<()> {
        let sorted_data = vec![1, 2, 3, 4, 5];
        let arr = ROCArray::from_vec(sorted_data)?;
        assert!(arr.is_sorted()?);

        let unsorted_data = vec![5, 2, 8, 1, 9];
        let arr = ROCArray::from_vec(unsorted_data)?;
        assert!(!arr.is_sorted()?);

        Ok(())
    }

    #[test]
    fn test_argsort() -> Result<()> {
        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        let arr = ROCArray::from_vec(data)?;

        let indices = arr.argsort()?;
        let result = indices.to_vec()?;

        // Indices should be [3, 1, 0, 2, 4] for ascending sort
        assert_eq!(result, vec![3, 1, 0, 2, 4]);

        Ok(())
    }

    #[test]
    fn test_empty_array_sort() -> Result<()> {
        let mut arr = ROCArray::<i32>::with_capacity(0)?;
        arr.sort()?;  // Should not panic
        assert_eq!(arr.len(), 0);

        Ok(())
    }

    #[test]
    fn test_single_element_sort() -> Result<()> {
        let mut arr = ROCArray::from_vec(vec![42])?;
        arr.sort()?;

        let result = arr.to_vec()?;
        assert_eq!(result, vec![42]);

        Ok(())
    }
}