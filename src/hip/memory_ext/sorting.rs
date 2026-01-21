use crate::hip::kernel::AsKernelArg;
use rocm_kernel_macros::{
    amdgpu_device, amdgpu_global, amdgpu_kernel_finalize, amdgpu_kernel_init,
};

amdgpu_kernel_init!(path: __build_in_kernels_sorting);

#[amdgpu_device(__build_in_kernels_sorting)]
use core::{cmp::PartialOrd, ptr::swap};

use crate::{
    hip::{DeviceMemory, Dim3, Module, Stream, error::Result},
    kernel_args,
};

#[amdgpu_device(__build_in_kernels_sorting)]
fn sort_odd_inner<T: Clone + Copy + PartialOrd>(arr: *mut T, ascending: bool) {
    let id_x = workgroup_id_x() as usize;

    let fst_index = id_x * 2 + 1;
    let sec_index = fst_index + 1;

    let fst = unsafe { *arr.add(fst_index) };
    let sec = unsafe { *arr.add(sec_index) };

    if (ascending && fst > sec) || (!ascending && fst < sec) {
        unsafe {
            swap(arr.add(fst_index), arr.add(sec_index));
        }
    }
}

#[amdgpu_device(__build_in_kernels_sorting)]
fn sort_even_inner<T: Clone + Copy + PartialOrd>(arr: *mut T, ascending: bool) {
    let id_x = workgroup_id_x() as usize;

    let fst_index = id_x * 2;
    let sec_index = fst_index + 1;

    let fst = unsafe { *arr.add(fst_index) };
    let sec = unsafe { *arr.add(sec_index) };

    if (ascending && fst > sec) || (!ascending && fst < sec) {
        unsafe {
            swap(arr.add(fst_index), arr.add(sec_index));
        }
    }
}

#[amdgpu_device(__build_in_kernels_sorting)]
fn check_sorted_inner<T: Clone + Copy + PartialOrd>(arr: *mut T, target: *mut bool, size: usize) {
    let id_x = workgroup_id_x() as usize;

    if (id_x >= size) {
        return;
    }

    let fst = unsafe { *arr.add(id_x) };
    let sec = unsafe { *arr.add(id_x + 1) };

    if (fst <= sec) {
        unsafe { *target.add(id_x) = true }
    } else {
        unsafe { *target.add(id_x) = false }
    }
}

macro_rules! sort_fns {
    ($t:ty) => {
        paste::paste! {
            #[amdgpu_global(__build_in_kernels_sorting)]
            fn [<sort_odd_$t>](arr: *mut $t, ascending: bool) {
                sort_odd_inner::<$t>(arr, ascending)
            }

            #[amdgpu_global(__build_in_kernels_sorting)]
            fn [<sort_even_$t>](arr: *mut $t, ascending: bool) {
                sort_even_inner::<$t>(arr, ascending)
            }

            #[amdgpu_global(__build_in_kernels_sorting)]
            fn [<check_sorted_$t>](arr: *mut $t, target: *mut bool, size: usize) {
                check_sorted_inner::<$t>(arr, target, size)
            }
        }
    };
}


macro_rules! generate_gpu_sort {
    ($($t:ty),+) => {
        $(
            sort_fns!($t);
        )*
    };
}

generate_gpu_sort!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

pub(crate) const SORTING_KERNEL: &[u8] =
    include_bytes!(amdgpu_kernel_finalize!(__build_in_kernels_sorting));

pub(crate) fn sort<T>(mem: &mut DeviceMemory<T>, stream: &Stream, ascending: bool) -> Result<()> {
    let module = Module::load_data(SORTING_KERNEL)?;

    let sort_odd =
        module.get_function(&(String::from("sort_odd_") + std::any::type_name::<T>()))?;
    let sort_even =
        module.get_function(&(String::from("sort_even_") + std::any::type_name::<T>()))?;

    let count = mem.count() as u32;

    let args = kernel_args!(mem, ascending);

    let grid_dim_even = Dim3::new_1d(count / 2);
    let grid_dim_odd = Dim3::new_1d((count - 1) / 2);

    for _ in 0..count / 2 {
        sort_even.launch(grid_dim_even, Dim3::new_1d(1), 0, Some(stream), args)?;
        sort_odd.launch(grid_dim_odd, Dim3::new_1d(1), 0, Some(stream), args)?;
    }

    Ok(())
}

/// Tis function synchronizes stream
///
/// This function will return an error if memory size is zero.
pub(crate) fn check_sorted<T>(mem: &DeviceMemory<T>, stream: Option<&Stream>) -> Result<bool> {
    let module = Module::load_data(SORTING_KERNEL)?;

    let check_sorted =
        module.get_function(&(String::from("check_sorted_") + std::any::type_name::<T>()))?;

    let count = mem.count();

    let target = DeviceMemory::<bool>::new(count - 1)?;

    let args = kernel_args!(mem, target, count);

    check_sorted.launch(
        Dim3::new_1d(count as u32 - 1),
        Dim3::new_1d(1),
        0,
        stream,
        args,
    )?;
    let mut host = vec![false; count - 1];
    if let Some(stream) = stream {
        let pending = target.copy_to_host_async(host, stream)?;
        host = stream.synchronize_memory(pending)?;
    } else {
        target.copy_to_host(&mut host)?;
    }
    Ok(host.iter().all(|x| *x))
}

#[cfg(test)]
mod test {
    use crate::{
        error::Result,
        hip::{
            Device, DeviceMemory,
            memory_ext::sorting::check_sorted,
        },
    };

    #[test]
    fn is_sorted() -> Result<()> {
        let device = Device::current()?;

        let stream = device.get_stream()?;

        let arr: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let mem = DeviceMemory::new(arr.len())?;
        mem.copy_from_host_async(arr, &stream)?;

        assert!(check_sorted(&mem, Some(&stream))?);

        Ok(())
    }

    #[test]
    fn is_not_sorted() -> Result<()> {
        let device = Device::current()?;

        let stream = device.get_stream()?;

        let arr: Vec<i32> = vec![1, 3, 2, 4, 5, 6, 8, 7];

        let mem = DeviceMemory::new(arr.len())?;
        mem.copy_from_host_async(arr, &stream)?;

        assert!(!check_sorted(&mem, Some(&stream))?);

        Ok(())
    }
}
