pub mod sorting;

use crate::hip::memory_ext::sorting::GPUSortAllowed;
use crate::hip::{DeviceMemory, Result, Stream};

pub trait MemoryExt<T> {
    fn sort(&mut self) -> Result<()>;
    fn sort_desc(&mut self) -> Result<()>;
    fn sort_async(&mut self, stream: &Stream) -> Result<()>;
    fn sort_desc_async(&mut self, stream: &Stream) -> Result<()>;
    fn check_sorted(&mut self) -> Result<bool>;
    fn check_sorted_async(&mut self, stream: &Stream) -> Result<bool>;
}

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

    fn check_sorted(&mut self) -> Result<bool> {
        sorting::check_sorted(self, None)    
    }

    fn check_sorted_async(&mut self, stream: &Stream) -> Result<bool> {
        sorting::check_sorted(self, Some(stream))
    }
}
