// src/rocrand/generator.rs
use std::ptr::{self, NonNull};

use crate::hip::DeviceMemory;
use crate::rocrand::bindings;
use crate::rocrand::error::{Error, Result};

/// Common trait for rocrand generators.
///
/// This trait defines operations common to all generators.
/// It is implemented for both pseudo-random and quasi-random generators.
pub trait Generator {
    /// Get the underlying raw generator pointer
    fn as_ptr(&self) -> bindings::rocrand_generator;

    /// Set the stream for kernel launches
    unsafe fn set_stream(&mut self, stream: bindings::hipStream_t) -> Result<()> {
        unsafe { Error::from_status(bindings::rocrand_set_stream(self.as_ptr(), stream)) }
    }

    /// Set the ordering of the generator
    fn set_ordering(&mut self, ordering: u32) -> Result<()> {
        unsafe { Error::from_status(bindings::rocrand_set_ordering(self.as_ptr(), ordering)) }
    }

    /// Initialize the generator
    fn initialize(&mut self) -> Result<()> {
        unsafe { Error::from_status(bindings::rocrand_initialize_generator(self.as_ptr())) }
    }

    /// Get the version of the rocrand library
    fn get_version() -> Result<i32> {
        let mut version = 0;
        unsafe {
            Error::from_status(bindings::rocrand_get_version(&mut version))?;
            Ok(version)
        }
    }
}

/// A pseudorandom number generator.
///
/// This struct wraps a rocrand generator and provides a safe interface to
/// generate various types of random numbers.
pub struct PseudoRng {
    generator: NonNull<bindings::rocrand_generator_base_type>,
}

impl PseudoRng {
    /// Create a new pseudorandom number generator of the specified type.
    ///
    /// # Examples
    ///
    /// ```
    /// use rocm_rs::rocrand::{PseudoRng, rng_type};
    ///
    /// let generator = PseudoRng::new(rng_type::XORWOW).unwrap();
    /// ```
    pub fn new(rng_type: u32) -> Result<Self> {
        let mut generator = ptr::null_mut();
        unsafe {
            Error::from_status(bindings::rocrand_create_generator(&mut generator, rng_type))?;
            Ok(Self {
                generator: NonNull::new(generator).unwrap(),
            })
        }
    }

    /// Create a new host-side pseudorandom number generator of the specified type.
    pub fn new_host(rng_type: u32) -> Result<Self> {
        let mut generator = ptr::null_mut();
        unsafe {
            Error::from_status(bindings::rocrand_create_generator_host(
                &mut generator,
                rng_type,
            ))?;
            Ok(Self {
                generator: NonNull::new(generator).unwrap(),
            })
        }
    }

    /// Set the seed for the generator.
    ///
    /// This operation resets the generator's internal state.
    /// This operation does not change the generator's offset.
    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        unsafe { Error::from_status(bindings::rocrand_set_seed(self.generator.as_ptr(), seed)) }
    }

    /// Set the seeds array for the generator (only for LFSR113).
    ///
    /// This operation resets the generator's internal state.
    /// This operation does not change the generator's offset.
    pub fn set_seed_array(&mut self, seed: u128) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_set_seed_uint4(
                self.generator.as_ptr(),
                seed,
            ))
        }
    }

    /// Set the absolute offset of the generator.
    ///
    /// This operation resets the generator's internal state.
    /// This operation does not change the generator's seed.
    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_set_offset(
                self.generator.as_ptr(),
                offset,
            ))
        }
    }

    /// Generate uniformly distributed 32-bit integers.
    ///
    /// Generated numbers are between 0 and 2^32-1.
    pub fn generate_u32(&mut self, output: &mut DeviceMemory<u32>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed 64-bit integers.
    ///
    /// Generated numbers are between 0 and 2^64-1.
    pub fn generate_u64(&mut self, output: &mut DeviceMemory<u64>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_long_long(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed 8-bit integers.
    ///
    /// Generated numbers are between 0 and 255.
    pub fn generate_u8(&mut self, output: &mut DeviceMemory<u8>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_char(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed 16-bit integers.
    ///
    /// Generated numbers are between 0 and 65535.
    pub fn generate_u16(&mut self, output: &mut DeviceMemory<u16>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_short(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed f32 values.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_uniform(&mut self, output: &mut DeviceMemory<f32>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_uniform(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed f64 values.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_uniform_double(&mut self, output: &mut DeviceMemory<f64>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_uniform_double(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate normally distributed f32 values.
    ///
    /// Generated numbers follow a normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate_normal(
        &mut self,
        output: &mut DeviceMemory<f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_normal(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
                mean,
                stddev,
            ))
        }
    }

    /// Generate normally distributed f64 values.
    ///
    /// Generated numbers follow a normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate_normal_double(
        &mut self,
        output: &mut DeviceMemory<f64>,
        mean: f64,
        stddev: f64,
    ) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_normal_double(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
                mean,
                stddev,
            ))
        }
    }

    /// Generate log-normally distributed f32 values.
    ///
    /// Generated numbers follow a log-normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate_log_normal(
        &mut self,
        output: &mut DeviceMemory<f32>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_log_normal(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
                mean,
                stddev,
            ))
        }
    }

    /// Generate log-normally distributed f64 values.
    ///
    /// Generated numbers follow a log-normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate_log_normal_double(
        &mut self,
        output: &mut DeviceMemory<f64>,
        mean: f64,
        stddev: f64,
    ) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_log_normal_double(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
                mean,
                stddev,
            ))
        }
    }

    /// Generate Poisson-distributed 32-bit integers.
    ///
    /// Generated numbers follow a Poisson distribution with the specified lambda.
    pub fn generate_poisson(&mut self, output: &mut DeviceMemory<u32>, lambda: f64) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_poisson(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
                lambda,
            ))
        }
    }
}

impl Generator for PseudoRng {
    fn as_ptr(&self) -> bindings::rocrand_generator {
        self.generator.as_ptr()
    }
}

impl Drop for PseudoRng {
    fn drop(&mut self) {
        unsafe {
            let _ = bindings::rocrand_destroy_generator(self.generator.as_ptr());
        }
    }
}

/// A quasirandom number generator.
///
/// This struct wraps a rocrand quasirandom generator and provides a safe interface to
/// generate various types of quasirandom numbers.
pub struct QuasiRng {
    generator: NonNull<bindings::rocrand_generator_base_type>,
}

impl QuasiRng {
    /// Create a new quasirandom number generator of the specified type.
    pub fn new(rng_type: u32) -> Result<Self> {
        let mut generator = ptr::null_mut();
        unsafe {
            Error::from_status(bindings::rocrand_create_generator(&mut generator, rng_type))?;
            Ok(Self {
                generator: NonNull::new(generator).unwrap(),
            })
        }
    }

    /// Set the number of dimensions for the generator.
    ///
    /// Supported values of dimensions are 1 to 20000.
    pub fn set_dimensions(&mut self, dimensions: u32) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_set_quasi_random_generator_dimensions(
                self.generator.as_ptr(),
                dimensions,
            ))
        }
    }

    /// Set the offset for the generator.
    ///
    /// This operation resets the generator's internal state.
    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_set_offset(
                self.generator.as_ptr(),
                offset,
            ))
        }
    }

    /// Generate uniformly distributed f32 values.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_uniform(&mut self, output: &mut DeviceMemory<f32>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_uniform(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }

    /// Generate uniformly distributed f64 values.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_uniform_double(&mut self, output: &mut DeviceMemory<f64>) -> Result<()> {
        unsafe {
            Error::from_status(bindings::rocrand_generate_uniform_double(
                self.generator.as_ptr(),
                output.as_ptr().cast(),
                output.count(),
            ))
        }
    }
}

impl Generator for QuasiRng {
    fn as_ptr(&self) -> bindings::rocrand_generator {
        self.generator.as_ptr()
    }
}

impl Drop for QuasiRng {
    fn drop(&mut self) {
        unsafe {
            let _ = bindings::rocrand_destroy_generator(self.generator.as_ptr());
        }
    }
}
