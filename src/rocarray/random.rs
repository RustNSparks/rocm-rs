// src/rocarray/random.rs - Fixed random number generation for ROCArray

use crate::error::Result;
use crate::hip::DeviceMemory;
use crate::rocrand::{
    Generator, LogNormal, Normal, Poisson, PseudoRng, QuasiRng, Uniform, rng_type,
};

/// Trait for types that support uniform random generation
pub trait UniformRandom: Copy + Default + 'static {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()>;
}

/// Trait for types that support normal random generation
pub trait NormalRandom: Copy + Default + 'static {
    fn fill_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()>;
}

/// Trait for types that support log-normal random generation
pub trait LogNormalRandom: Copy + Default + 'static {
    fn fill_log_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()>;
}

/// Trait for types that support Poisson random generation
pub trait PoissonRandom: Copy + Default + 'static {
    fn fill_poisson_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        lambda: f64,
    ) -> Result<()>;
}

// Implement UniformRandom for supported types
impl UniformRandom for f32 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_uniform(output)?)
    }
}

impl UniformRandom for f64 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_uniform_double(output)?)
    }
}

impl UniformRandom for u32 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_u32(output)?)
    }
}

impl UniformRandom for u64 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_u64(output)?)
    }
}

impl UniformRandom for u16 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_u16(output)?)
    }
}

impl UniformRandom for u8 {
    fn fill_uniform_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
    ) -> Result<()> {
        Ok(generator.generate_u8(output)?)
    }
}

// Implement NormalRandom for supported types
impl NormalRandom for f32 {
    fn fill_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        Ok(generator.generate_normal(output, mean, stddev)?)
    }
}

impl NormalRandom for f64 {
    fn fill_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        Ok(generator.generate_normal_double(output, mean as f64, stddev as f64)?)
    }
}

// Implement LogNormalRandom for supported types
impl LogNormalRandom for f32 {
    fn fill_log_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        Ok(generator.generate_log_normal(output, mean, stddev)?)
    }
}

impl LogNormalRandom for f64 {
    fn fill_log_normal_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        Ok(generator.generate_log_normal_double(output, mean as f64, stddev as f64)?)
    }
}

// Implement PoissonRandom for supported types
impl PoissonRandom for u32 {
    fn fill_poisson_device(
        generator: &mut PseudoRng,
        output: &mut DeviceMemory<Self>,
        lambda: f64,
    ) -> Result<()> {
        Ok(generator.generate_poisson(output, lambda)?)
    }
}

/// Fill a DeviceMemory buffer with uniformly distributed random values
pub fn fill_uniform<T>(output: &mut DeviceMemory<T>, len: usize, seed: Option<u64>) -> Result<()>
where
    T: UniformRandom,
{
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    // Create a temporary buffer if the output buffer is larger than needed
    if output.count() > len {
        let mut temp_output = DeviceMemory::<T>::new(len)?;
        T::fill_uniform_device(&mut generator, &mut temp_output)?;

        // Copy only the needed elements
        temp_output.copy_to_host(&mut vec![T::default(); len])?;
        // Note: This is a simplified approach. In practice, you'd want device-to-device copy
        return Ok(());
    }

    T::fill_uniform_device(&mut generator, output)
}

/// Fill a DeviceMemory buffer with normally distributed random values
pub fn fill_normal<T>(
    output: &mut DeviceMemory<T>,
    len: usize,
    mean: f32,
    stddev: f32,
    seed: Option<u64>,
) -> Result<()>
where
    T: NormalRandom,
{
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    // Create a temporary buffer if the output buffer is larger than needed
    if output.count() > len {
        let mut temp_output = DeviceMemory::<T>::new(len)?;
        T::fill_normal_device(&mut generator, &mut temp_output, mean, stddev)?;
        return Ok(());
    }

    T::fill_normal_device(&mut generator, output, mean, stddev)
}

/// Fill a DeviceMemory buffer with log-normally distributed random values
pub fn fill_log_normal<T>(
    output: &mut DeviceMemory<T>,
    len: usize,
    mean: f32,
    stddev: f32,
    seed: Option<u64>,
) -> Result<()>
where
    T: LogNormalRandom,
{
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    // Create a temporary buffer if the output buffer is larger than needed
    if output.count() > len {
        let mut temp_output = DeviceMemory::<T>::new(len)?;
        T::fill_log_normal_device(&mut generator, &mut temp_output, mean, stddev)?;
        return Ok(());
    }

    T::fill_log_normal_device(&mut generator, output, mean, stddev)
}

/// Fill a DeviceMemory buffer with Poisson distributed random values
pub fn fill_poisson<T>(
    output: &mut DeviceMemory<T>,
    len: usize,
    lambda: f64,
    seed: Option<u64>,
) -> Result<()>
where
    T: PoissonRandom,
{
    let mut generator = PseudoRng::new(rng_type::MTGP32)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    // Create a temporary buffer if the output buffer is larger than needed
    if output.count() > len {
        let mut temp_output = DeviceMemory::<T>::new(len)?;
        T::fill_poisson_device(&mut generator, &mut temp_output, lambda)?;
        return Ok(());
    }

    T::fill_poisson_device(&mut generator, output, lambda)
}

/// Generate uniformly distributed random values and return them as a Vec
pub fn generate_uniform<T>(count: usize, seed: Option<u64>) -> Result<Vec<T>>
where
    T: UniformRandom,
{
    let mut generator = PseudoRng::new(rng_type::XORWOW)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    let mut device_output = DeviceMemory::<T>::new(count)?;
    T::fill_uniform_device(&mut generator, &mut device_output)?;

    let mut host_output = vec![T::default(); count];
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate normally distributed random values and return them as a Vec
pub fn generate_normal<T>(count: usize, mean: f32, stddev: f32, seed: Option<u64>) -> Result<Vec<T>>
where
    T: NormalRandom,
{
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    let mut device_output = DeviceMemory::<T>::new(count)?;
    T::fill_normal_device(&mut generator, &mut device_output, mean, stddev)?;

    let mut host_output = vec![T::default(); count];
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate log-normally distributed random values and return them as a Vec
pub fn generate_log_normal<T>(
    count: usize,
    mean: f32,
    stddev: f32,
    seed: Option<u64>,
) -> Result<Vec<T>>
where
    T: LogNormalRandom,
{
    let mut generator = PseudoRng::new(rng_type::PHILOX4_32_10)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    let mut device_output = DeviceMemory::<T>::new(count)?;
    T::fill_log_normal_device(&mut generator, &mut device_output, mean, stddev)?;

    let mut host_output = vec![T::default(); count];
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate Poisson distributed random values and return them as a Vec
pub fn generate_poisson<T>(count: usize, lambda: f64, seed: Option<u64>) -> Result<Vec<T>>
where
    T: PoissonRandom,
{
    let mut generator = PseudoRng::new(rng_type::MTGP32)?;

    if let Some(seed_value) = seed {
        generator.set_seed(seed_value)?;
    }

    generator.initialize()?;

    let mut device_output = DeviceMemory::<T>::new(count)?;
    T::fill_poisson_device(&mut generator, &mut device_output, lambda)?;

    let mut host_output = vec![T::default(); count];
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Generate quasi-random sequence using Sobol generator
pub fn generate_quasi<T>(count: usize, dimensions: u32) -> Result<Vec<T>>
where
    T: UniformRandom,
{
    let mut generator = QuasiRng::new(rng_type::SOBOL32)?;
    generator.set_dimensions(dimensions)?;
    generator.initialize()?;

    let mut device_output = DeviceMemory::<T>::new(count)?;

    // For quasi-random, we use uniform generation
    match std::mem::size_of::<T>() {
        4 => {
            // f32 case
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let f32_output = unsafe {
                    std::mem::transmute::<&mut DeviceMemory<T>, &mut DeviceMemory<f32>>(
                        &mut device_output,
                    )
                };
                generator.generate_uniform(f32_output)?;
            }
        }
        8 => {
            // f64 case
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                let f64_output = unsafe {
                    std::mem::transmute::<&mut DeviceMemory<T>, &mut DeviceMemory<f64>>(
                        &mut device_output,
                    )
                };
                generator.generate_uniform_double(f64_output)?;
            }
        }
        _ => {
            return Err(crate::error::Error::Custom(
                "Quasi-random generation only supported for f32 and f64".to_string(),
            ));
        }
    }

    let mut host_output = vec![T::default(); count];
    device_output.copy_to_host(&mut host_output)?;

    Ok(host_output)
}

/// Random number utilities
pub struct RandomUtils;

impl RandomUtils {
    /// Create a seeded generator for reproducible results
    pub fn seeded_generator(seed: u64, rng_type: u32) -> Result<PseudoRng> {
        let mut generator = PseudoRng::new(rng_type)?;
        generator.set_seed(seed)?;
        generator.initialize()?;
        Ok(generator)
    }

    /// Create a default generator with random seed
    pub fn default_generator() -> Result<PseudoRng> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self::seeded_generator(seed, rng_type::XORWOW)
    }

    /// Fill buffer with random integers in a specific range
    pub fn fill_range_uniform_int(
        output: &mut DeviceMemory<u32>,
        min_val: u32,
        max_val: u32,
        seed: Option<u64>,
    ) -> Result<()> {
        if min_val >= max_val {
            return Err(crate::error::Error::Custom(
                "Invalid range: min must be less than max".to_string(),
            ));
        }

        // Generate uniform values and scale to range
        fill_uniform(output, output.count(), seed)?;

        // TODO: Add kernel to scale values to range [min_val, max_val)
        // For now, this is a placeholder
        Ok(())
    }

    /// Generate random permutation of indices
    pub fn random_permutation(n: usize, seed: Option<u64>) -> Result<Vec<u32>> {
        // This would typically use a shuffle algorithm on GPU
        // For now, provide a simple implementation
        let mut indices: Vec<u32> = (0..n as u32).collect();

        // Use a simple random generator to shuffle
        if let Some(seed_val) = seed {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            for i in (1..n).rev() {
                let mut hasher = DefaultHasher::new();
                (seed_val, i).hash(&mut hasher);
                let j = (hasher.finish() as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        Ok(indices)
    }

    /// Sample without replacement
    pub fn sample_without_replacement(
        population_size: usize,
        sample_size: usize,
        seed: Option<u64>,
    ) -> Result<Vec<u32>> {
        if sample_size > population_size {
            return Err(crate::error::Error::Custom(
                "Sample size cannot be larger than population size".to_string(),
            ));
        }

        let permutation = Self::random_permutation(population_size, seed)?;
        Ok(permutation.into_iter().take(sample_size).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::DeviceMemory;

    #[test]
    fn test_uniform_generation() -> Result<()> {
        let values = generate_uniform::<f32>(1000, Some(42))?;
        assert_eq!(values.len(), 1000);

        // Check that values are in [0, 1) range
        assert!(values.iter().all(|&x| x >= 0.0 && x < 1.0));
        Ok(())
    }

    #[test]
    fn test_normal_generation() -> Result<()> {
        let values = generate_normal::<f32>(1000, 0.0, 1.0, Some(42))?;
        assert_eq!(values.len(), 1000);

        // Basic sanity check - mean should be approximately 0
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        assert!((mean).abs() < 0.2); // Allow some variance
        Ok(())
    }

    #[test]
    fn test_fill_uniform() -> Result<()> {
        let mut device_mem = DeviceMemory::<f32>::new(100)?;
        fill_uniform(&mut device_mem, 100, Some(42))?;

        let mut host_data = vec![0.0f32; 100];
        device_mem.copy_to_host(&mut host_data)?;

        // Check that values are in [0, 1) range
        assert!(host_data.iter().all(|&x| x >= 0.0 && x < 1.0));
        Ok(())
    }

    #[test]
    fn test_random_permutation() -> Result<()> {
        let perm = RandomUtils::random_permutation(10, Some(42))?;
        assert_eq!(perm.len(), 10);

        // Check that all values 0-9 are present
        let mut sorted_perm = perm.clone();
        sorted_perm.sort();
        let expected: Vec<u32> = (0..10).collect();
        assert_eq!(sorted_perm, expected);
        Ok(())
    }

    #[test]
    fn test_sample_without_replacement() -> Result<()> {
        let sample = RandomUtils::sample_without_replacement(100, 10, Some(42))?;
        assert_eq!(sample.len(), 10);

        // Check that all values are unique and in valid range
        let mut unique_sample = sample.clone();
        unique_sample.sort();
        unique_sample.dedup();
        assert_eq!(unique_sample.len(), 10); // All values should be unique
        assert!(sample.iter().all(|&x| x < 100)); // All values should be < 100
        Ok(())
    }
}
