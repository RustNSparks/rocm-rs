// src/rocarray/sorting.rs - Complete implementation
use crate::error::Result;
use crate::hip::kernel::AsKernelArg;
use crate::hip::memory_ext::sorting::GPUSortAllowed;
use crate::hip::{
    DeviceMemory, Dim3, Function, Module, Stream, calculate_grid_1d, memory_ext::MemoryExt,
};
use std::sync::Once;

static INIT_SORT: Once = Once::new();
static mut SORT_MODULE: Option<Module> = None;

pub trait Sortable: Copy + Default + PartialOrd + 'static {
    const TYPE_NAME: &'static str;
}

impl Sortable for i32 {
    const TYPE_NAME: &'static str = "int";
}

impl Sortable for f32 {
    const TYPE_NAME: &'static str = "float";
}

impl Sortable for f64 {
    const TYPE_NAME: &'static str = "double";
}

impl Sortable for u32 {
    const TYPE_NAME: &'static str = "uint";
}

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
            Ok(module.get_function(name)?)
        } else {
            Err(crate::error::Error::InvalidOperation(
                "Sort kernels not initialized".to_string(),
            ))
        }
    }
}

// Ascending sort
pub fn sort_ascending<T: GPUSortAllowed>(data: &mut DeviceMemory<T>) -> Result<()> {
    data.sort().map_err(|err| err.into())
}

pub fn sort_ascending_async<T: GPUSortAllowed>(
    data: &mut DeviceMemory<T>,
    stream: &Stream,
) -> Result<()> {
    data.sort_async(stream).map_err(|err| err.into())
}

// Descending sort
pub fn sort_descending<T: GPUSortAllowed>(data: &mut DeviceMemory<T>) -> Result<()> {
    data.sort_desc().map_err(|err| err.into())
}

pub fn sort_descending_async<T: GPUSortAllowed>(
    data: &mut DeviceMemory<T>,
    stream: &Stream,
) -> Result<()> {
    data.sort_desc_async(stream).map_err(|err| err.into())
}

// Argsort - returns indices that would sort the array
pub fn argsort<T>(data: &DeviceMemory<T>, indices: &DeviceMemory<u32>, len: usize) -> Result<()>
where
    T: Sortable,
{
    let stream = Stream::new()?;
    argsort_async(data, indices, len, &stream)?;
    stream.synchronize()?;
    Ok(())
}

pub fn argsort_async<T>(
    data: &DeviceMemory<T>,
    indices: &DeviceMemory<u32>,
    len: usize,
    stream: &Stream,
) -> Result<()>
where
    T: Sortable,
{
    if len <= 1 {
        return Ok(());
    }

    // First, initialize indices to 0, 1, 2, ...
    let init_kernel = get_sort_kernel_function("init_indices")?;
    let block_size = 256;
    let grid_dim = calculate_grid_1d(len as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    let len_u32 = len as u32;
    let init_args = [
        indices.as_kernel_arg(),
        &len_u32 as *const _ as *mut std::ffi::c_void,
    ];

    init_kernel.launch(grid_dim, block_dim, 0, Some(stream), &mut init_args.clone())?;

    // Then sort indices based on data values
    let kernel_name = format!("argsort_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let sort_args = [
        data.as_kernel_arg(),
        indices.as_kernel_arg(),
        &len_u32 as *const _ as *mut std::ffi::c_void,
    ];

    function.launch(grid_dim, block_dim, 0, Some(stream), &mut sort_args.clone())?;
    Ok(())
}

// Check if array is sorted
pub fn is_sorted<T: GPUSortAllowed>(data: &DeviceMemory<T>) -> Result<bool> {
    data.check_sorted().map_err(|err| err.into())
}

// Partial sort (sort only the first k elements)
pub fn partial_sort<T>(data: &mut DeviceMemory<T>, len: usize, k: usize) -> Result<()>
where
    T: Sortable + GPUSortAllowed,
{
    if k >= len {
        return sort_ascending(data);
    }

    let stream = Stream::new()?;
    let kernel_name = format!("partial_sort_{}", T::TYPE_NAME);
    let function = get_sort_kernel_function(&kernel_name)?;

    let grid_dim = Dim3::new_1d(1);
    let block_dim = Dim3::new_1d(1);

    let len_u32 = len as u32;
    let k_u32 = k as u32;
    let kernel_args = [
        data.as_kernel_arg(),
        &len_u32 as *const _ as *mut std::ffi::c_void,
        &k_u32 as *const _ as *mut std::ffi::c_void,
    ];

    function.launch(
        grid_dim,
        block_dim,
        0,
        Some(&stream),
        &mut kernel_args.clone(),
    )?;
    stream.synchronize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocarray::ROCArray;

    #[test]
    fn test_sort_ascending() -> Result<()> {
        let data = vec![5, 2, 8, 1, 9, 3];
        let mut arr = ROCArray::from_vec(data)?;

        arr.sort()?;
        let result = arr.to_vec()?;
        assert_eq!(result, vec![1, 2, 3, 5, 8, 9]);
        Ok(())
    }

    #[test]
    fn test_sort_descending() -> Result<()> {
        let data = vec![5, 2, 8, 1, 9, 3];
        let mut arr = ROCArray::from_vec(data)?;

        arr.sort_descending()?;
        let result = arr.to_vec()?;
        assert_eq!(result, vec![9, 8, 5, 3, 2, 1]);
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
    fn test_partial_sort() -> Result<()> {
        let data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        let mut arr = ROCArray::from_vec(data)?;
        let len = arr.len();

        // Sort only the first 3 elements
        partial_sort(&mut arr.data, len, 3)?;

        let result = arr.to_vec()?;
        // First 3 elements should be the 3 smallest: [1, 2, 3]
        assert_eq!(&result[0..3], &[1, 2, 3]);
        Ok(())
    }
}
