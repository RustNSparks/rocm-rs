extern crate core;
pub mod rocrand;
pub mod rocfft;
pub mod miopen;
pub mod rocblas;
pub mod rocsolver;

pub mod hip;
pub mod error;
use crate::rocfft::examples;

#[cfg(feature="rocm_smi")]
pub mod rocmsmi;

#[cfg(test)]
mod tests {
    use crate::rocfft::examples::run_1d_complex_example;

    #[test]
    fn test_rocrand() {
        run_1d_complex_example().unwrap();
        run_1d_complex_example().unwrap();
    }
}