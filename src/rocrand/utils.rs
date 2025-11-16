// src/rocrand/utils.rs
//
// Utility functions for easier use of the rocrand library

use crate::error::Result;
use crate::hip::DeviceMemory;
use crate::rocrand::{
    Generator, LogNormal, Normal, Poisson, PseudoRng, QuasiRng, Uniform, rng_type,
}; // Using our unified error type

macro_rules! generate_uniform_rand_func {
    ($fn_name: ident, $data_type:ty, $generato_fn:ident, $rng_type:ident) => {
        paste::paste! {
            #[doc = "Generate random " $data_type " values on device"]
            pub fn $fn_name(
                count: usize,
                seed: Option<u64>,
            ) -> Result<DeviceMemory<$data_type>> {
                // Create a generator
                let mut generator = PseudoRng::new(rng_type::$rng_type)?;
                // Set seed if provided
                if let Some(seed_value) = seed {
                    generator.set_seed(seed_value)?;
                }
                // Initialize the generator
                generator.initialize()?;
                // Allocate device memory
                let mut device_output = DeviceMemory::<$data_type>::new(count)?;

                // Generate the random numbers
                generator.$generato_fn(&mut device_output)?;

                Ok(device_output)
            }
        }
    };
}

generate_uniform_rand_func!(generate_uniform_f32, f32, generate_uniform, XORWOW);
generate_uniform_rand_func!(generate_uniform_f64, f64, generate_uniform_double, XORWOW);
generate_uniform_rand_func!(generate_u32, u32, generate_u32, XORWOW);

macro_rules! generate_normal_rand_func {
    ($fn_name: ident, $data_type:ty, $rng_type:ident, $dist:expr) => {
        paste::paste! {
            #[doc = "Generate normally distributed random " $data_type " values with specified mean and standard deviation"]
            pub fn $fn_name(
                count: usize,
                mean: f32,
                stddev: f32,
                seed: Option<u64>,
            ) -> Result<DeviceMemory<$data_type>> {
                // Create a generator
                let mut generator = PseudoRng::new(rng_type::$rng_type)?;

                // Set seed if provided
                if let Some(seed_value) = seed {
                    generator.set_seed(seed_value)?;
                }

                // Initialize the generator
                generator.initialize()?;

                // Create a normal distribution
                let dist = $dist(mean, stddev);

                // Allocate device memory
                let mut device_output = DeviceMemory::<f32>::new(count)?;

                // Generate the random numbers
                dist.generate(&mut generator, &mut device_output)?;

                Ok(device_output)
            }
        }
    };
}

generate_normal_rand_func!(generate_normal_f32, f32, PHILOX4_32_10, Normal::new);
generate_normal_rand_func!(generate_log_normal_f32, f32, PHILOX4_32_10, LogNormal::new);

/// Generate Poisson-distributed random u32 values with specified lambda
pub fn generate_poisson(count: usize, lambda: f64, seed: Option<u64>) -> Result<DeviceMemory<u32>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::MTGP32)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Create a poisson distribution
    let poisson_dist = Poisson::new(lambda);

    // Allocate device memory
    let mut device_output = DeviceMemory::<u32>::new(count)?;

    // Generate the random numbers
    poisson_dist.generate(&mut generator, &mut device_output)?;

    Ok(device_output)
}

/// Generate quasirandom sequence of f32 values with specified dimensions
pub fn generate_quasi_f32(count: usize, dimensions: u32) -> Result<DeviceMemory<f32>> {
    // Create a quasi-random generator
    let mut generator = QuasiRng::new(rng_type::SOBOL32)?;

    // Set dimensions
    generator.set_dimensions(dimensions)?;

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Generate the random numbers
    Uniform::generate_quasi(&mut generator, &mut device_output)?;

    Ok(device_output)
}