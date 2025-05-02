// examples/rocfft_examples.rs

use crate::hip::{self, Device, DeviceMemory, Stream};
use crate::rocfft::{self, plan::{Plan, PlacementType, Precision, TransformType, ArrayType}, description::PlanDescription};

/// Example of a 1D complex-to-complex FFT
pub fn run_1d_complex_example() -> Result<(), Box<dyn std::error::Error>> {
    // Set up a simple 1D complex FFT
    let length = 1024;
    let lengths = vec![length];
    
    // Create a plan for a forward complex FFT
    let mut plan = Plan::new(
        PlacementType::InPlace,
        TransformType::ComplexForward,
        Precision::Single,
        1,
        &lengths,
        1,
        None,
    )?;
    
    // Create input data (complex interleaved: real, imag, real, imag, ...)
    let complex_length = length * 2;
    let mut input_data = vec![0.0f32; complex_length];
    
    // Initialize with a sine wave - ENSURE THIS IS WORKING
    println!("Initializing sine wave...");
    for i in 0..length {
        let x = i as f32 / length as f32;
        // A strong sine wave signal at frequency 10
        input_data[i * 2] = f32::sin(2.0 * std::f32::consts::PI * 10.0 * x);
        input_data[i * 2 + 1] = 0.0; // Imaginary part is zero
    }
    
    // Print a few input values to verify
    println!("Input data samples:");
    for i in 0..5 {
        println!("  Element {}: Real={:.4}, Imag={:.4}", 
                i, input_data[i*2], input_data[i*2+1]);
    }
    
    // Allocate device memory
    let mut d_input = DeviceMemory::<f32>::new(complex_length)?;
    
    // Copy input data to device
    d_input.copy_from_host(&input_data)?;
    
    // Execute the transform
    let input_ptr = [d_input.as_ptr()];
    plan.execute(&input_ptr, &[], None)?;
    
    // Copy result back to host
    let mut output_data = vec![0.0f32; complex_length];
    d_input.copy_to_host(&mut output_data)?;
    
    // Print results
    println!("FFT Results:");
    let mut found_nonzero = false;
    
    // Print the first few frequency bins
    for i in 0..15 {
        let real = output_data[i * 2];
        let imag = output_data[i * 2 + 1];
        let magnitude = (real * real + imag * imag).sqrt();
        println!("Freq {}: Magnitude = {:.4}", i, magnitude);
        
        if magnitude > 0.1 {
            found_nonzero = true;
        }
    }
    
    if !found_nonzero {
        println!("WARNING: No significant magnitudes found. This may indicate a problem.");
        
        // Check some more frequencies, including bin 10 where we expect a peak
        println!("Checking bin 10 where we expect a peak:");
        let i = 10;
        let real = output_data[i * 2];
        let imag = output_data[i * 2 + 1];
        let magnitude = (real * real + imag * imag).sqrt();
        println!("Freq {}: Magnitude = {:.4}", i, magnitude);
    }
    
    Ok(())
}
/// Example of a 1D real-to-complex and complex-to-real FFT
/// Demonstrates FFT-based filtering
pub fn run_1d_real_example() -> Result<(), Box<dyn std::error::Error>> {
    // Set up a 1D real FFT
    let length = 1024;
    let lengths = vec![length];

    // For real-to-complex transform, output size is (length/2 + 1) due to conjugate symmetry
    let complex_length = (length / 2 + 1) * 2; // *2 for complex data (real, imag pairs)

    // Create a plan for a forward real-to-complex FFT
    let mut forward_plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::RealForward,
        Precision::Single,
        1,
        &lengths,
        1,
        None,
    )?;

    // Create a plan description for the inverse transform with scaling
    let mut description = PlanDescription::new()?;
    description.set_scale_factor(1.0 / length as f64)?; // Scale by 1/N

    // Create a plan for an inverse complex-to-real FFT
    let mut inverse_plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::RealInverse,
        Precision::Single,
        1,
        &lengths,
        1,
        Some(&description),
    )?;

    // Create real input data
    let mut input_data = vec![0.0f32; length];

    // Initialize input with a simple sine wave plus noise
    for i in 0..length {
        let x = i as f32 / length as f32;
        input_data[i] = f32::sin(2.0 * std::f32::consts::PI * 10.0 * x) +
            0.5 * f32::sin(2.0 * std::f32::consts::PI * 25.0 * x);
    }

    // Allocate device memory
    let mut d_input = DeviceMemory::<f32>::new(length)?;
    let mut d_output = DeviceMemory::<f32>::new(complex_length)?;
    let mut d_reconstructed = DeviceMemory::<f32>::new(length)?;

    // Copy input data to device
    d_input.copy_from_host(&input_data)?;

    // Execute the forward transform (real to complex)
    let input_ptr = [d_input.as_ptr()];
    let output_ptr = [d_output.as_ptr()];
    forward_plan.execute(&input_ptr, &output_ptr, None)?;

    // Copy result back to host
    let mut output_data = vec![0.0f32; complex_length];
    d_output.copy_to_host(&mut output_data)?;

    // Print frequency domain data
    println!("Real FFT Results (first 5 frequencies):");
    for i in 0..5 {
        let mag = (output_data[i*2].powi(2) + output_data[i*2+1].powi(2)).sqrt();
        println!("Freq {}: Magnitude = {:.4}", i, mag);
    }

    // Filter the signal (zero out high frequencies as an example)
    // Keep only the first 20% of frequencies
    let cutoff = (length / 2 + 1) / 5;
    for i in cutoff..(length/2+1) {
        output_data[i*2] = 0.0;     // Real part
        output_data[i*2+1] = 0.0;   // Imaginary part
    }

    // Copy filtered data back to device
    d_output.copy_from_host(&output_data)?;

    // Execute the inverse transform (complex to real)
    let inverse_input_ptr = [d_output.as_ptr()];
    let inverse_output_ptr = [d_reconstructed.as_ptr()];
    inverse_plan.execute(&inverse_input_ptr, &inverse_output_ptr, None)?;

    // Copy result back to host
    let mut reconstructed_data = vec![0.0f32; length];
    d_reconstructed.copy_to_host(&mut reconstructed_data)?;

    // Compare original and reconstructed data
    let mut error_sum = 0.0;
    for i in 0..length {
        error_sum += (input_data[i] - reconstructed_data[i]).abs();
    }
    let avg_error = error_sum / length as f32;

    println!("Average error after low-pass filtering: {:.4e}", avg_error);

    Ok(())
}