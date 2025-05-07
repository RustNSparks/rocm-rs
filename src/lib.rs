extern crate core;
pub mod miopen;
pub mod rocblas;
pub mod rocfft;
pub mod rocrand;
pub mod rocsolver;

pub mod error;
pub mod hip;
use crate::rocfft::examples;

#[cfg(feature = "rocm_smi")]
pub mod rocmsmi;
pub mod rocsparse;

#[cfg(test)]
mod tests {
    use crate::rocfft::examples::run_1d_complex_example;

    #[test]
    fn test_rocrand() {
        run_1d_complex_example().unwrap();
        run_1d_complex_example().unwrap();
    }
}
