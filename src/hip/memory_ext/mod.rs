#[cfg(feature = "macros")]
pub mod sorting;

use crate::hip::{DeviceMemory, Result, Stream};

pub trait MemoryExt<T> {
    fn sort(&mut self) -> Result<()>;
    fn sort_desc(&mut self) -> Result<()>;
    fn sort_async(&mut self, stream: &Stream) -> Result<()>;
    fn sort_desc_async(&mut self, stream: &Stream) -> Result<()>;
    fn check_sorted(&self) -> Result<bool>;
    fn check_sorted_async(&self, stream: &Stream) -> Result<bool>;
}

pub trait GPUSortAllowed {}
macro_rules! impl_gpu_sort_allowed {
    ($($t:ty),+) => {
        $(
            impl GPUSortAllowed for $t {}
        )*
    };
}
impl_gpu_sort_allowed!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

#[cfg(feature = "macros")]
impl<T> MemoryExt<T> for DeviceMemory<T>
where
    T: GPUSortAllowed,
{
    fn sort(&mut self) -> Result<()> {
        let stream = Stream::new()?;
        self.sort_async(&stream)?;
        stream.synchronize()?;
        Ok(())
    }

    fn sort_desc(&mut self) -> Result<()> {
        let stream = Stream::new()?;
        self.sort_desc_async(&stream)?;
        stream.synchronize()?;
        Ok(())
    }

    fn sort_async(&mut self, stream: &Stream) -> Result<()> {
        sorting::sort(self, stream, true)
    }

    fn sort_desc_async(&mut self, stream: &Stream) -> Result<()> {
        sorting::sort(self, stream, false)
    }

    fn check_sorted(&self) -> Result<bool> {
        sorting::check_sorted(self, None)
    }

    fn check_sorted_async(&self, stream: &Stream) -> Result<bool> {
        sorting::check_sorted(self, Some(stream))
    }
}

#[cfg(not(feature = "macros"))]
impl<T> MemoryExt<T> for DeviceMemory<T>
where
    T: GPUSortAllowed,
{
    fn sort(&mut self) -> Result<()> {
        println!(
            "WARNING: sort function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }

    fn sort_desc(&mut self) -> Result<()> {
        println!(
            "WARNING: sort_desc function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }

    fn sort_async(&mut self, stream: &Stream) -> Result<()> {
        println!(
            "WARNING: sort_async function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }

    fn sort_desc_async(&mut self, stream: &Stream) -> Result<()> {
        println!(
            "WARNING: sort_desc_async function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }

    fn check_sorted(&self) -> Result<bool> {
        println!(
            "WARNING: this function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }

    fn check_sorted_async(&self, stream: &Stream) -> Result<bool> {
        println!(
            "WARNING: check_sorted_async function is not implemented because the 'macros' feature is disabled."
        );
        return Err(crate::hip::Error::new(
            crate::hip::ffi::hipError_t_hipErrorNotInitialized,
        ));
    }
}
