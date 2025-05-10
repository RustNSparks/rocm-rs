// src/rocrand/mod.rs
//
// Module definition for rocrand

// Re-export the raw bindings for advanced usage
#[allow(warnings)]
pub mod bindings;

// Import submodules
pub mod distribution;
pub mod error;
pub mod generator;
pub mod utils;

// Re-export public items
pub use distribution::{Discrete, LogNormal, Normal, Poisson, Uniform};
pub use error::{Error, Result};
pub use generator::{Generator, PseudoRng, QuasiRng};

/// Convenient re-exports of random number generator types
pub mod rng_type {
    use super::bindings;

    pub const PSEUDO_DEFAULT: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_DEFAULT;
    pub const XORWOW: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_XORWOW;
    pub const MRG32K3A: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_MRG32K3A;
    pub const MTGP32: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_MTGP32;
    pub const PHILOX4_32_10: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
    pub const MRG31K3P: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_MRG31K3P;
    pub const LFSR113: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_LFSR113;
    pub const MT19937: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_MT19937;
    pub const THREEFRY2_32_20: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_THREEFRY2_32_20;
    pub const THREEFRY2_64_20: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_THREEFRY2_64_20;
    pub const THREEFRY4_32_20: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_THREEFRY4_32_20;
    pub const THREEFRY4_64_20: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_PSEUDO_THREEFRY4_64_20;

    pub const QUASI_DEFAULT: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_QUASI_DEFAULT;
    pub const SOBOL32: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_QUASI_SOBOL32;
    pub const SCRAMBLED_SOBOL32: u32 =
        bindings::rocrand_rng_type_ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    pub const SOBOL64: u32 = bindings::rocrand_rng_type_ROCRAND_RNG_QUASI_SOBOL64;
    pub const SCRAMBLED_SOBOL64: u32 =
        bindings::rocrand_rng_type_ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

/// Convenient re-exports of ordering constants
pub mod ordering {
    use super::bindings;

    pub const PSEUDO_BEST: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_PSEUDO_BEST;
    pub const PSEUDO_DEFAULT: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_PSEUDO_DEFAULT;
    pub const PSEUDO_SEEDED: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_PSEUDO_SEEDED;
    pub const PSEUDO_LEGACY: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_PSEUDO_LEGACY;
    pub const PSEUDO_DYNAMIC: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_PSEUDO_DYNAMIC;
    pub const QUASI_DEFAULT: u32 = bindings::rocrand_ordering_ROCRAND_ORDERING_QUASI_DEFAULT;
}

/// Re-export direction vector constants
pub mod direction_vector_set {
    use super::bindings;

    pub const VECTORS_32_JOEKUO6: u32 =
        bindings::rocrand_direction_vector_set_ROCRAND_DIRECTION_VECTORS_32_JOEKUO6;
    pub const SCRAMBLED_VECTORS_32_JOEKUO6: u32 =
        bindings::rocrand_direction_vector_set_ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
    pub const VECTORS_64_JOEKUO6: u32 =
        bindings::rocrand_direction_vector_set_ROCRAND_DIRECTION_VECTORS_64_JOEKUO6;
    pub const SCRAMBLED_VECTORS_64_JOEKUO6: u32 =
        bindings::rocrand_direction_vector_set_ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
}

/// Creates the default pseudo-random number generator
pub fn default_generator() -> Result<PseudoRng> {
    PseudoRng::new(rng_type::PSEUDO_DEFAULT)
}

/// Creates a XORWOW pseudo-random number generator
pub fn xorwow_generator() -> Result<PseudoRng> {
    PseudoRng::new(rng_type::XORWOW)
}

/// Creates a Sobol32 quasi-random number generator with the specified dimensions
pub fn sobol32_generator(dimensions: u32) -> Result<QuasiRng> {
    let mut rng = QuasiRng::new(rng_type::SOBOL32)?;
    rng.set_dimensions(dimensions)?;
    Ok(rng)
}

/// Gets the rocRAND library version
pub fn get_version() -> Result<i32> {
    // Use fully qualified syntax to call the trait function
    <PseudoRng as Generator>::get_version()
}
