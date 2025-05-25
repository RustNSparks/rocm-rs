// src/rocrand/distribution.rs

use crate::hip::DeviceMemory;
use crate::rocrand::bindings;
use crate::rocrand::error::{Error, Result};
use crate::rocrand::generator::{PseudoRng, QuasiRng};
use std::ptr::NonNull;

/// Uniform distribution for generating values in range [0, 1).
pub struct Uniform;

impl Uniform {
    /// Generate uniformly distributed f32 values using a pseudo-random generator.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate(generator: &mut PseudoRng, output: &mut DeviceMemory<f32>) -> Result<()> {
        generator.generate_uniform(output)
    }

    /// Generate uniformly distributed f64 values using a pseudo-random generator.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_double(generator: &mut PseudoRng, output: &mut DeviceMemory<f64>) -> Result<()> {
        generator.generate_uniform_double(output)
    }

    /// Generate uniformly distributed f32 values using a quasi-random generator.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_quasi(generator: &mut QuasiRng, output: &mut DeviceMemory<f32>) -> Result<()> {
        generator.generate_uniform(output)
    }

    /// Generate uniformly distributed f64 values using a quasi-random generator.
    ///
    /// Generated numbers are between 0.0 and 1.0.
    pub fn generate_quasi_double(generator: &mut QuasiRng, output: &mut DeviceMemory<f64>) -> Result<()> {
        generator.generate_uniform_double(output)
    }
}

/// Normal (Gaussian) distribution.
pub struct Normal {
    mean: f32,
    stddev: f32,
}

impl Normal {
    /// Create a new normal distribution with the specified mean and standard deviation.
    pub fn new(mean: f32, stddev: f32) -> Self {
        Self { mean, stddev }
    }

    /// Generate normally distributed f32 values.
    ///
    /// Generated numbers follow a normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate(&self, generator: &mut PseudoRng, output: &mut DeviceMemory<f32>) -> Result<()> {
        generator.generate_normal(output, self.mean, self.stddev)
    }
}

/// Normal (Gaussian) distribution with f64 precision.
pub struct NormalDouble {
    mean: f64,
    stddev: f64,
}

impl NormalDouble {
    /// Create a new normal distribution with the specified mean and standard deviation.
    pub fn new(mean: f64, stddev: f64) -> Self {
        Self { mean, stddev }
    }

    /// Generate normally distributed f64 values.
    ///
    /// Generated numbers follow a normal distribution with the specified
    /// mean and standard deviation.
    pub fn generate(&self, generator: &mut PseudoRng, output: &mut DeviceMemory<f64>) -> Result<()> {
        generator.generate_normal_double(output, self.mean, self.stddev)
    }
}

/// Log-normal distribution.
pub struct LogNormal {
    mean: f32,
    stddev: f32,
}

impl LogNormal {
    /// Create a new log-normal distribution with the specified mean and standard deviation.
    pub fn new(mean: f32, stddev: f32) -> Self {
        Self { mean, stddev }
    }

    /// Generate log-normally distributed f32 values.
    pub fn generate(&self, generator: &mut PseudoRng, output: &mut DeviceMemory<f32>) -> Result<()> {
        generator.generate_log_normal(output, self.mean, self.stddev)
    }
}

/// Log-normal distribution with f64 precision.
pub struct LogNormalDouble {
    mean: f64,
    stddev: f64,
}

impl LogNormalDouble {
    /// Create a new log-normal distribution with the specified mean and standard deviation.
    pub fn new(mean: f64, stddev: f64) -> Self {
        Self { mean, stddev }
    }

    /// Generate log-normally distributed f64 values.
    pub fn generate(&self, generator: &mut PseudoRng, output: &mut DeviceMemory<f64>) -> Result<()> {
        generator.generate_log_normal_double(output, self.mean, self.stddev)
    }
}

/// Poisson distribution.
pub struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Create a new Poisson distribution with the specified lambda.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Generate Poisson-distributed integers.
    pub fn generate(&self, generator: &mut PseudoRng, output: &mut DeviceMemory<u32>) -> Result<()> {
        generator.generate_poisson(output, self.lambda)
    }
}

/// Discrete distribution for generating custom probability distributions.
pub struct Discrete {
    distribution: NonNull<bindings::rocrand_discrete_distribution_st>,
}

impl Discrete {
    /// Create a discrete distribution from an array of probabilities.
    ///
    /// The probabilities array specifies the relative probability of each value,
    /// starting from `offset`.
    pub fn from_probabilities(probabilities: &[f64], offset: u32) -> Result<Self> {
        let mut distribution = std::ptr::null_mut();
        unsafe {
            Error::from_status(bindings::rocrand_create_discrete_distribution(
                probabilities.as_ptr(),
                probabilities.len() as u32,
                offset,
                &mut distribution,
            ))?;
            Ok(Self {
                distribution: NonNull::new(distribution).unwrap(),
            })
        }
    }

    /// Create a discrete distribution corresponding to the Poisson distribution.
    ///
    /// This creates a discrete distribution histogram that approximates a Poisson
    /// distribution with the given lambda.
    pub fn poisson(lambda: f64) -> Result<Self> {
        let mut distribution = std::ptr::null_mut();
        unsafe {
            Error::from_status(bindings::rocrand_create_poisson_distribution(
                lambda,
                &mut distribution,
            ))?;
            Ok(Self {
                distribution: NonNull::new(distribution).unwrap(),
            })
        }
    }

    /// Get the raw pointer to the distribution.
    pub fn as_ptr(&self) -> bindings::rocrand_discrete_distribution {
        self.distribution.as_ptr()
    }
}

impl Drop for Discrete {
    fn drop(&mut self) {
        unsafe {
            let _ = bindings::rocrand_destroy_discrete_distribution(self.distribution.as_ptr());
        }
    }
}
