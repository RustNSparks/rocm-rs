// src/rocfft/utils.rs

use crate::error::Error::RocFFT;
use crate::error::Result;
use crate::hip::{DeviceMemory, Stream};
use crate::rocfft::{
    description::PlanDescription,
    error,
    execution::ExecutionInfo,
    plan::{ArrayType, PlacementType, Plan, Precision, TransformType},
};

/// Determines the size of the output for a real-to-complex transform
///
/// For a real FFT, the output size in the first dimension is `(input_size / 2) + 1`
/// due to conjugate symmetry, but unchanged in other dimensions.
///
/// # Arguments
/// * `input_lengths` - The sizes of each dimension in the input array
///
/// # Returns
/// * The sizes of each dimension in the output array
pub fn get_real_forward_output_length(input_lengths: &[usize]) -> Vec<usize> {
    let mut output_lengths = input_lengths.to_vec();

    if !output_lengths.is_empty() {
        // For real-to-complex FFT, output first dimension is (n/2)+1
        output_lengths[0] = (input_lengths[0] / 2) + 1;
    }

    output_lengths
}

/// Wrapper for forward complex-to-complex FFT
///
/// # Arguments
/// * `input` - Input data on GPU (complex interleaved)
/// * `output` - Optional output data on GPU (for out-of-place transform)
/// * `lengths` - Size of each dimension
/// * `precision` - Numerical precision
/// * `stream` - Optional GPU stream for execution
///
/// # Returns
/// * Result indicating success or error
pub unsafe fn complex_forward_transform<T>(
    input: &DeviceMemory<T>,
    output: Option<&DeviceMemory<T>>,
    lengths: &[usize],
    precision: Precision,
    stream: Option<&Stream>,
) -> Result<()> {
    let placement = match output {
        Some(_) => PlacementType::NotInPlace,
        None => PlacementType::InPlace,
    };

    let dimensions = lengths.len();
    if dimensions < 1 || dimensions > 3 {
        return Err(RocFFT(error::Error::InvalidDimensions));
    }

    // Create default plan
    let mut plan = Plan::new(
        placement,
        TransformType::ComplexForward,
        precision,
        dimensions,
        lengths,
        1, // Single transform
        None,
    )?;

    // Set up execution info if we have a stream
    let mut exec_info = match stream {
        Some(s) => {
            let mut info = ExecutionInfo::new()?;
            unsafe { info.set_stream(s.as_raw() as *mut std::ffi::c_void) }?;
            Some(info)
        }
        None => None,
    };

    // Setup input/output buffers
    let input_ptr = [input.as_ptr()];
    let output_ptrs = match output {
        Some(out) => vec![out.as_ptr()],
        None => vec![],
    };

    // Execute the plan
    plan.execute(&input_ptr, &output_ptrs, exec_info.as_mut())?;

    Ok(())
}

/// Wrapper for inverse complex-to-complex FFT
///
/// # Arguments
/// * `input` - Input data on GPU (complex interleaved)
/// * `output` - Optional output data on GPU (for out-of-place transform)
/// * `lengths` - Size of each dimension
/// * `precision` - Numerical precision
/// * `scale` - Whether to apply 1/N scaling
/// * `stream` - Optional GPU stream for execution
///
/// # Returns
/// * Result indicating success or error
pub fn complex_inverse_transform<T>(
    input: &DeviceMemory<T>,
    output: Option<&DeviceMemory<T>>,
    lengths: &[usize],
    precision: Precision,
    scale: bool,
    stream: Option<&Stream>,
) -> Result<()> {
    let placement = match output {
        Some(_) => PlacementType::NotInPlace,
        None => PlacementType::InPlace,
    };

    let dimensions = lengths.len();
    if dimensions < 1 || dimensions > 3 {
        return Err(RocFFT(error::Error::InvalidDimensions));
    }

    // Calculate total number of elements to determine scaling factor
    let total_elements: usize = lengths.iter().product();
    let scale_factor = if scale {
        1.0 / total_elements as f64
    } else {
        1.0
    };

    // Create plan description if we need scaling
    let description = if scale {
        let mut desc = PlanDescription::new()?;
        desc.set_scale_factor(scale_factor)?;
        Some(desc)
    } else {
        None
    };

    // Create plan
    let mut plan = Plan::new(
        placement,
        TransformType::ComplexInverse,
        precision,
        dimensions,
        lengths,
        1, // Single transform
        description.as_ref(),
    )?;

    // Set up execution info if we have a stream
    let mut exec_info = match stream {
        Some(s) => unsafe {
            let mut info = ExecutionInfo::new()?;
            info.set_stream(s.as_raw() as *mut std::ffi::c_void)?;
            Some(info)
        },
        None => None,
    };

    // Setup input/output buffers
    let input_ptr = [input.as_ptr()];
    let output_ptrs = match output {
        Some(out) => vec![out.as_ptr()],
        None => vec![],
    };

    // Execute the plan
    plan.execute(&input_ptr, &output_ptrs, exec_info.as_mut())?;

    Ok(())
}

/// Wrapper for forward real-to-complex FFT
///
/// # Arguments
/// * `input` - Input real data on GPU
/// * `output` - Output complex data on GPU
/// * `lengths` - Size of each dimension of input
/// * `precision` - Numerical precision
/// * `stream` - Optional GPU stream for execution
///
/// # Returns
/// * Result indicating success or error
pub fn real_forward_transform<T, U>(
    input: &DeviceMemory<T>,  // Real input
    output: &DeviceMemory<U>, // Complex output (interleaved)
    lengths: &[usize],
    precision: Precision,
    stream: Option<&Stream>,
) -> Result<()> {
    let dimensions = lengths.len();
    if dimensions < 1 || dimensions > 3 {
        return Err(RocFFT(error::Error::InvalidDimensions));
    }

    // Create default plan for real-to-complex transform
    let mut plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::RealForward,
        precision,
        dimensions,
        lengths,
        1, // Single transform
        None,
    )?;

    // Set up execution info if we have a stream
    let mut exec_info = match stream {
        Some(s) => unsafe {
            let mut info = ExecutionInfo::new()?;
            info.set_stream(s.as_raw() as *mut std::ffi::c_void)?;
            Some(info)
        },
        None => None,
    };

    // Setup input/output buffers
    let input_ptr = [input.as_ptr()];
    let output_ptr = [output.as_ptr()];

    // Execute the plan
    plan.execute(&input_ptr, &output_ptr, exec_info.as_mut())?;

    Ok(())
}

/// Wrapper for inverse complex-to-real FFT
///
/// # Arguments
/// * `input` - Input complex data on GPU
/// * `output` - Output real data on GPU
/// * `lengths` - Size of each dimension of the output
/// * `precision` - Numerical precision
/// * `scale` - Whether to apply 1/N scaling
/// * `stream` - Optional GPU stream for execution
///
/// # Returns
/// * Result indicating success or error
pub fn complex_to_real_transform<T, U>(
    input: &DeviceMemory<T>,  // Complex input (interleaved)
    output: &DeviceMemory<U>, // Real output
    lengths: &[usize],
    precision: Precision,
    scale: bool,
    stream: Option<&Stream>,
) -> Result<()> {
    let dimensions = lengths.len();
    if dimensions < 1 || dimensions > 3 {
        return Err(RocFFT(error::Error::InvalidDimensions));
    }

    // Calculate total number of elements to determine scaling factor
    let total_elements: usize = lengths.iter().product();
    let scale_factor = if scale {
        1.0 / total_elements as f64
    } else {
        1.0
    };

    // Create plan description if we need scaling
    let description = if scale {
        let mut desc = PlanDescription::new()?;
        desc.set_scale_factor(scale_factor)?;
        Some(desc)
    } else {
        None
    };

    // Create plan
    let mut plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::RealInverse,
        precision,
        dimensions,
        lengths, // Pass the full output lengths
        1,       // Single transform
        description.as_ref(),
    )?;

    // Set up execution info if we have a stream
    let mut exec_info = match stream {
        Some(s) => unsafe {
            let mut info = ExecutionInfo::new()?;
            info.set_stream(s.as_raw() as *mut std::ffi::c_void)?;
            Some(info)
        },
        None => None,
    };

    // Setup input/output buffers
    let input_ptr = [input.as_ptr()];
    let output_ptr = [output.as_ptr()];

    // Execute the plan
    plan.execute(&input_ptr, &output_ptr, exec_info.as_mut())?;

    Ok(())
}

/// Create and initialize 2D FFT plan with custom strides and offsets
///
/// # Arguments
/// * `width` - Width of the transform (columns)
/// * `height` - Height of the transform (rows)
/// * `in_row_stride` - Stride between elements in the same row
/// * `in_col_stride` - Stride between elements in the same column
/// * `transform_type` - Type of transform
/// * `precision` - Numerical precision
/// * `placement` - In-place or out-of-place
///
/// # Returns
/// * FFT plan with the specified parameters
pub fn create_2d_fft_plan_with_strides(
    width: usize,
    height: usize,
    in_row_stride: usize,
    in_col_stride: usize,
    transform_type: TransformType,
    precision: Precision,
    placement: PlacementType,
) -> Result<Plan> {
    let lengths = vec![width, height];
    let dimensions = 2;

    // Create a plan description for the custom strides
    let mut description = PlanDescription::new()?;

    // Set strides (row stride is distance between adjacent elements,
    // column stride is distance between rows)
    let in_strides = vec![in_row_stride, in_col_stride];

    // If real-to-complex transform, output size is different
    let (in_array_type, out_array_type) = match transform_type {
        TransformType::RealForward => (ArrayType::Real, ArrayType::ComplexInterleaved),
        TransformType::RealInverse => (ArrayType::ComplexInterleaved, ArrayType::Real),
        _ => (ArrayType::ComplexInterleaved, ArrayType::ComplexInterleaved),
    };

    // Calculate distance between batches (total size of one 2D array)
    let in_distance = height * in_col_stride;

    // For simplicity, we use the same strides for output
    // In a real application, these might differ
    let out_strides = in_strides.clone();
    let out_distance = in_distance;

    // Set data layout
    description.set_data_layout(
        in_array_type,
        out_array_type,
        Some(&[0]), // No offset
        Some(&[0]), // No offset
        Some(&in_strides),
        in_distance,
        Some(&out_strides),
        out_distance,
    )?;

    // Create the plan
    let plan = Plan::new(
        placement,
        transform_type,
        precision,
        dimensions,
        &lengths,
        1, // Single transform
        Some(&description),
    )?;

    Ok(plan)
}

/// Utility function to apply a 1D convolution using FFT
///
/// # Arguments
/// * `signal` - Input signal on device
/// * `kernel` - Convolution kernel on device
/// * `output` - Output buffer (must be same size as signal)
/// * `precision` - Numerical precision
/// * `stream` - Optional GPU stream
///
/// # Returns
/// * Result indicating success or error
pub fn fft_convolution_1d<T>(
    signal: &DeviceMemory<T>,
    kernel: &DeviceMemory<T>,
    output: &mut DeviceMemory<T>,
    precision: Precision,
    stream: Option<&Stream>,
) -> Result<()>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::ops::Neg<Output = T> + std::ops::Add<Output = T>,
{
    // Create work buffers for FFT
    let signal_size = signal.count();
    let kernel_size = kernel.count();

    if signal_size < kernel_size {
        return Err(RocFFT(error::Error::InvalidArgValue));
    }

    // Create padded work buffers (signal length + kernel length - 1)
    let padded_size = signal_size + kernel_size - 1;
    let lengths = vec![padded_size];

    // Allocate device memory for padded input and kernel
    let mut padded_signal = DeviceMemory::<T>::new(padded_size * 2)?; // *2 for complex
    let mut padded_kernel = DeviceMemory::<T>::new(padded_size * 2)?; // *2 for complex
    let mut fft_result = DeviceMemory::<T>::new(padded_size * 2)?; // *2 for complex

    // Create plans
    let mut forward_plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::ComplexForward,
        precision,
        1, // 1D
        &lengths,
        1, // Single transform
        None,
    )?;

    // Create inverse plan with scaling
    let mut desc = PlanDescription::new()?;
    desc.set_scale_factor(1.0 / padded_size as f64)?;

    let mut inverse_plan = Plan::new(
        PlacementType::NotInPlace,
        TransformType::ComplexInverse,
        precision,
        1, // 1D
        &lengths,
        1, // Single transform
        Some(&desc),
    )?;

    // Create execution info if we have a stream
    let mut exec_info = match stream {
        Some(s) => unsafe {
            let mut info = ExecutionInfo::new()?;
            info.set_stream(s.as_raw() as *mut std::ffi::c_void)?;
            Some(info)
        },
        None => None,
    };

    // Host buffers for preparing data
    let mut host_padded_signal = vec![T::default(); padded_size * 2];
    let mut host_padded_kernel = vec![T::default(); padded_size * 2];

    // Copy signal to host buffer (interleaved complex: real, imag, real, imag, ...)
    let mut host_signal = vec![T::default(); signal_size];
    signal.copy_to_host(&mut host_signal)?;

    for i in 0..signal_size {
        host_padded_signal[i * 2] = host_signal[i]; // Real part
        // Imaginary part is already initialized to zero
    }

    // Copy kernel to host buffer
    let mut host_kernel = vec![T::default(); kernel_size];
    kernel.copy_to_host(&mut host_kernel)?;

    for i in 0..kernel_size {
        host_padded_kernel[i * 2] = host_kernel[i]; // Real part
        // Imaginary part is already initialized to zero
    }

    // Copy data to device
    padded_signal.copy_from_host(&host_padded_signal)?;
    padded_kernel.copy_from_host(&host_padded_kernel)?;

    // Perform forward FFTs
    let input_ptr = [padded_signal.as_ptr()];
    let kernel_ptr = [padded_kernel.as_ptr()];
    let result_ptr = [fft_result.as_ptr()];

    // FFT of signal
    forward_plan.execute(&input_ptr, &result_ptr, exec_info.as_mut())?;

    // Copy result to padded_signal for reuse
    let mut host_fft_signal = vec![T::default(); padded_size * 2];
    fft_result.copy_to_host(&mut host_fft_signal)?;

    // FFT of kernel
    forward_plan.execute(&kernel_ptr, &result_ptr, exec_info.as_mut())?;

    // Copy kernel FFT result
    let mut host_fft_kernel = vec![T::default(); padded_size * 2];
    fft_result.copy_to_host(&mut host_fft_kernel)?;

    // Perform pointwise multiplication in frequency domain
    let mut host_mult_result = vec![T::default(); padded_size * 2];

    // Simple example assuming T is f32 or f64
    // In a real implementation you'd need to handle complex multiplication properly
    // This is a simplification!
    for i in 0..padded_size {
        let idx = i * 2;
        let s_real = host_fft_signal[idx];
        let s_imag = host_fft_signal[idx + 1];
        let k_real = host_fft_kernel[idx];
        let k_imag = host_fft_kernel[idx + 1];

        // Complex multiplication (s_real + i*s_imag) * (k_real + i*k_imag)
        // This assumes T can be multiplied and added, which may not be true for all types
        // In a real implementation, you'd need proper complex number handling
        host_mult_result[idx] = multiply_add(s_real, k_real, multiply_neg(s_imag, k_imag)); // Real part
        host_mult_result[idx + 1] = multiply_add(s_real, k_imag, multiply(s_imag, k_real)); // Imaginary part
    }

    // Copy multiplication result back to device
    fft_result.copy_from_host(&host_mult_result)?;

    // Create a buffer for the inverse FFT result
    let ifft_result = DeviceMemory::<T>::new(padded_size * 2)?;
    let ifft_ptr = [ifft_result.as_ptr()];

    // Perform inverse FFT
    inverse_plan.execute(&result_ptr, &ifft_ptr, exec_info.as_mut())?;

    // Copy result back to host
    let mut host_ifft_result = vec![T::default(); padded_size * 2];
    ifft_result.copy_to_host(&mut host_ifft_result)?;

    // Extract real part of the result (first signal_size elements)
    let mut host_output = vec![T::default(); signal_size];
    for i in 0..signal_size {
        host_output[i] = host_ifft_result[i * 2]; // Take only real part
    }

    // Copy result to output
    output.copy_from_host(&host_output)?;

    Ok(())
}

// These helper functions would need proper implementations for the generic type T
// Here we just use placeholders

fn multiply<T: std::ops::Mul<Output = T>>(a: T, b: T) -> T {
    a*b
}

fn multiply_neg<T: std::ops::Mul<Output = T> + std::ops::Neg<Output = T>>(a: T, b: T) -> T {
    -multiply(a, b)
}

fn multiply_add<T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(a: T, b: T, c: T) -> T {
    multiply(a, b) + c
}
