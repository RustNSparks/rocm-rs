extern crate core;

pub mod hip;
pub mod rocfft;
pub mod miopen;
pub mod rocblas;
pub mod rocsolver;
pub mod rocsparse;
pub mod rocrand;

// write some tests for fft 

#[cfg(test)]
mod tests {
    use std::fmt::LowerHex;
    use crate::rocfft;
    use crate::rocfft::error::check_dimensions;

    #[test]
    fn test_fft() {
        match check_dimensions(1) { 
            Ok(..) => {
                println!("fft");
            }
            Err(_) => {
                panic!("fft failed");
            }
        }
    }
}