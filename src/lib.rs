extern crate core;
pub mod rocrand;
pub mod rocfft;
pub mod miopen;
pub mod rocblas;

pub mod hip;
pub mod error;

#[cfg(test)]
mod tests {
    use crate::hip::{get_device_count, host_mem_flags, Device, DeviceMemory, PinnedMemory};
    use crate::hip::utils::print_devices_info;
    use crate::rocblas::Handle;
    use crate::rocrand;
    use crate::rocrand::{rng_type, PseudoRng};
    use crate::rocrand::utils::generate_uniform_f32;

    #[test]
    fn test_rocrand() {
        let handle = Handle::new();
        match handle {
            Ok(handle) => {
                let device_count = get_device_count().unwrap();
                println!("Number of devices: {}", device_count);;
            }
            Err(e) => {
                eprintln!("Error creating handle: {}", e);
            }
        }
    }
}