pub mod sorting;


use crate::hip::kernel::AsKernelArg;
use crate::hip::memory_ext::sorting::{GPUSortAllowed, SORTING_KERNEL};
use crate::hip::{DeviceMemory, Dim3, Module, Result};

use crate::kernel_args;

pub trait MemoryExt<T> {
    fn sort(&mut self) -> Result<()>;
    fn sort_desc(&mut self) -> Result<()>;
}

impl<T> MemoryExt<T> for DeviceMemory<T>
where
    T: GPUSortAllowed,
{
    fn sort(&mut self) -> Result<()> {
        let module = Module::load_data(SORTING_KERNEL)?;

        let sort_odd =
            module.get_function(&(String::from("sort_odd_") + std::any::type_name::<T>()))?;
        let sort_even =
            module.get_function(&(String::from("sort_even_") + std::any::type_name::<T>()))?;

        let count = self.count() as u32;

        let args = kernel_args!(self, true);

        let dim_even = Dim3::new_1d(count / 2);
        let dim_odd = Dim3::new_1d((count - 1) / 2);

        for _ in 0..count / 2 {
            sort_even.launch(dim_even, Dim3::new_1d(1), 0, None, args)?;
            sort_odd.launch(dim_odd, Dim3::new_1d(1), 0, None, args)?;
        }

        Ok(())
    }

    fn sort_desc(&mut self) -> Result<()> {
        let module = Module::load_data(SORTING_KERNEL)?;

        let sort_odd =
            module.get_function(&(String::from("sort_odd_") + std::any::type_name::<T>()))?;
        let sort_even =
            module.get_function(&(String::from("sort_even_") + std::any::type_name::<T>()))?;

        let count = self.count() as u32;

        let args = kernel_args!(self, false);

        let dim_even = Dim3::new_1d(count / 2);
        let dim_odd = Dim3::new_1d((count - 1) / 2);

        for _ in 0..count / 2 {
            sort_even.launch(dim_even, Dim3::new_1d(1), 0, None, args)?;
            sort_odd.launch(dim_odd, Dim3::new_1d(1), 0, None, args)?;
        }

        Ok(())
    }
}
