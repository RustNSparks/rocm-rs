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
    use crate::rocfft::get_version;

    #[test]
    fn test_fft() {
        match get_version() {
            Ok(v) => {
                println!("{}", v)
            }
            Err(e) => panic!("{}", e),
        }
    }
}