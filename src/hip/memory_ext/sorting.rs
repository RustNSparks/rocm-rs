use rocm_kernel_macros::{
    amdgpu_device, amdgpu_global, amdgpu_kernel_finalize, amdgpu_kernel_init,
};

amdgpu_kernel_init!(path: __build_in_kernels_sorting);

#[amdgpu_device(__build_in_kernels_sorting)]
use core::{cmp::PartialOrd, ptr::swap};

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
        }
    };
}

pub trait GPUSortAllowed {}

macro_rules! impl_gpu_sort_allowed {
    ($($t:ty),+) => {
        $(
            impl GPUSortAllowed for $t {}
            sort_fns!($t);
        )*
    };
}

impl_gpu_sort_allowed!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

pub(crate) const SORTING_KERNEL: &[u8] =
    include_bytes!(amdgpu_kernel_finalize!(__build_in_kernels_sorting));
