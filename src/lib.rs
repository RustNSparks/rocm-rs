extern crate core;
pub mod rocrand;
pub mod rocfft;

pub mod hip;


#[cfg(test)]
mod tests {
    use crate::rocrand;
    use crate::rocrand::{rng_type, PseudoRng};

    #[test]
    fn test_rocrand() {
        let mut generator = PseudoRng::new(rng_type::XORWOW).unwrap();

        // Optional: Set a seed for reproducibility
        generator.set_seed(12345).unwrap();

        // Create an array to hold the random numbers
        let mut random_numbers: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Generate uniform random numbers between 0.0 and 1.0
        generator.generate_uniform(&mut random_numbers).unwrap();

        // Use the random numbers
        println!("First few random numbers: {:?}", &random_numbers[..5]);
        
    }
}