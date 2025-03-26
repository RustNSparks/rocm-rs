// src/rocrand/utils.rs
//
// Utility functions for easier use of the rocrand library

use crate::hip::{DeviceMemory, Stream};
use crate::rocrand::{PseudoRng, QuasiRng, rng_type, Uniform, Normal, LogNormal, Poisson, Generator};
use crate::error::{Result, Error}; // Using our unified error type

/// Generate uniformly distributed random f32 values on device and return them on host
pub fn generate_uniform_f32(count: usize, seed: Option<u64>) -> Result<Vec<f32>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate uniform random numbers
        generator.generate_uniform(slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate uniformly distributed random f64 values on device and return them on host
pub fn generate_uniform_f64(count: usize, seed: Option<u64>) -> Result<Vec<f64>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<f64>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f64; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f64;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate uniform random numbers
        generator.generate_uniform_double(slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate random u32 integers on device and return them on host
pub fn generate_u32(count: usize, seed: Option<u64>) -> Result<Vec<u32>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<u32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0u32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut u32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate random integers
        generator.generate_u32(slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate normally distributed random f32 values with specified mean and standard deviation
pub fn generate_normal_f32(count: usize, mean: f32, stddev: f32, seed: Option<u64>) -> Result<Vec<f32>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Create a normal distribution
    let normal_dist = Normal::new(mean, stddev);

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate normal random numbers
        normal_dist.generate(&mut generator, slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate log-normally distributed random f32 values with specified mean and standard deviation
pub fn generate_log_normal_f32(count: usize, mean: f32, stddev: f32, seed: Option<u64>) -> Result<Vec<f32>> {
    // Create a generator
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    // Set seed if provided
    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    // Initialize the generator
    generator.initialize()?;

    // Create a log-normal distribution
    let log_normal_dist = LogNormal::new(mean, stddev);

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate log-normal random numbers
        log_normal_dist.generate(&mut generator, slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate Poisson-distributed random u32 values with specified lambda
pub fn generate_poisson(count: usize, lambda: f64, seed: Option<u64>) -> Result<Vec<u32>> {
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

    // Create host output buffer
    let mut host_output = vec![0u32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut u32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate poisson random numbers
        poisson_dist.generate(&mut generator, slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate quasirandom sequence of f32 values with specified dimensions
pub fn generate_quasi_f32(count: usize, dimensions: u32) -> Result<Vec<f32>> {
    // Create a quasi-random generator
    let mut generator = QuasiRng::new(rng_type::SOBOL32)?;

    // Set dimensions
    generator.set_dimensions(dimensions)?;

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Generate quasi-random numbers
        Uniform::generate_quasi(&mut generator, slice)?;
    }

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate random values using a stream for asynchronous operation
pub fn generate_async<F>(count: usize, generator_fn: F) -> Result<Vec<f32>>
where
    F: FnOnce(&mut PseudoRng, &mut [f32]) -> std::result::Result<(), crate::rocrand::Error>
{
    // Create a stream
    let stream = Stream::new()?;

    // Create generator
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    // Set the stream
    generator.set_stream(stream.as_raw() as crate::rocrand::bindings::hipStream_t)?;

    // Initialize the generator
    generator.initialize()?;

    // Allocate device memory
    let mut device_output = DeviceMemory::<f32>::new(count)?;

    // Create host output buffer
    let mut host_output = vec![0.0f32; count];

    // Generate the random numbers
    unsafe {
        let device_ptr = device_output.as_ptr() as *mut f32;
        let slice = std::slice::from_raw_parts_mut(device_ptr, count);

        // Use the provided generator function
        generator_fn(&mut generator, slice)?;
    }

    // Synchronize the stream
    stream.synchronize()?;

    // Copy results back to host
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Fill a device memory buffer with random values directly
/// Note: This doesn't copy back to host, so the caller is responsible for that
pub fn fill_device_memory<T, F>(device_memory: &mut DeviceMemory<T>, generator_fn: F) -> Result<()>
where
    F: FnOnce(*mut T, usize) -> std::result::Result<(), crate::rocrand::Error>
{
    // Skip if empty
    if device_memory.count() == 0 {
        return Ok(());
    }

    // Generate the random numbers
    unsafe {
        let device_ptr = device_memory.as_ptr() as *mut T;
        generator_fn(device_ptr, device_memory.count())?;
    }

    Ok(())
}

/// Convenience function for creating a simple random buffer by type
pub enum RandomType {
    Uniform,
    Normal { mean: f32, stddev: f32 },
    LogNormal { mean: f32, stddev: f32 },
    Poisson { lambda: f64 },
    Integer,
}

pub fn create_random_buffer<T>(count: usize, random_type: RandomType, seed: Option<u64>) -> Result<Vec<T>>
where
    T: Copy + Default
{
    match (random_type, std::mem::size_of::<T>()) {
        // Handle f32 types
        (RandomType::Uniform, 4) => {
            let data = generate_uniform_f32(count, seed)?;
            let ptr = data.as_ptr() as *const T;
            unsafe {
                Ok(std::slice::from_raw_parts(ptr, count).to_vec())
            }
        },
        (RandomType::Normal { mean, stddev }, 4) => {
            let data = generate_normal_f32(count, mean, stddev, seed)?;
            let ptr = data.as_ptr() as *const T;
            unsafe {
                Ok(std::slice::from_raw_parts(ptr, count).to_vec())
            }
        },
        (RandomType::LogNormal { mean, stddev }, 4) => {
            let data = generate_log_normal_f32(count, mean, stddev, seed)?;
            let ptr = data.as_ptr() as *const T;
            unsafe {
                Ok(std::slice::from_raw_parts(ptr, count).to_vec())
            }
        },
        // Handle u32 types
        (RandomType::Integer, 4) => {
            let data = generate_u32(count, seed)?;
            let ptr = data.as_ptr() as *const T;
            unsafe {
                Ok(std::slice::from_raw_parts(ptr, count).to_vec())
            }
        },
        (RandomType::Poisson { lambda }, 4) => {
            let data = generate_poisson(count, lambda, seed)?;
            let ptr = data.as_ptr() as *const T;
            unsafe {
                Ok(std::slice::from_raw_parts(ptr, count).to_vec())
            }
        },
        // Handle other cases
        _ => {
            // Return a default buffer for unsupported types
            Ok(vec![T::default(); count])
        }
    }
}