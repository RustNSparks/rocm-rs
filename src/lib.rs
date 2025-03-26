extern crate core;
pub mod rocrand;
pub mod rocfft;
pub mod miopen;

pub mod hip;
pub mod error;

#[cfg(test)]
mod tests {
    use crate::hip::{get_device_count, host_mem_flags, Device, DeviceMemory, PinnedMemory};
    use crate::hip::utils::print_devices_info;
    use crate::rocrand;
    use crate::rocrand::{rng_type, PseudoRng};
    use crate::rocrand::utils::generate_uniform_f32;

    #[test]
    fn test_rocrand() {
        let device_memory = match DeviceMemory::<f32>::new(1000) {
            Ok(memory) => memory,
            Err(e) => panic!("{:?}", e),
        };

        // rocRAND operation (returns rocrand::Error on failure)
        let random_data = generate_uniform_f32(1000, Some(42)).unwrap();
        println!("random_data: {:?}", random_data);
        
    }
}