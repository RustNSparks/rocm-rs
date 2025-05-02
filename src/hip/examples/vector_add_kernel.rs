/// Example demonstrating launching a HIP kernel
use rocm_rs::error::Result;
use rocm_rs::hip::{
    Device, DeviceMemory, Stream, calculate_grid_1d, Dim3,
    kernel_launcher, device_synchronize
};

// This is just to demonstrate the macro. In a real application,
// you'd typically have this defined in a .hip file.
extern "C" {
    fn vector_add(a: *const f32, b: *const f32, c: *mut f32, n: u32);
}

// Use the macro to generate a launcher function
kernel_launcher!(launch_vector_add, vector_add, *const f32, *const f32, *mut f32, u32);

fn main() -> Result<()> {
    // Initialize device
    println!("Initializing device...");
    let device = Device::new(0)?;
    device.set_current()?;
    
    // Create a stream
    let stream = Stream::new()?;
    
    // Set up data
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
    let mut c = vec![0.0f32; n];
    
    // Allocate device memory
    let mut d_a = DeviceMemory::<f32>::new(n)?;
    let mut d_b = DeviceMemory::<f32>::new(n)?;
    let mut d_c = DeviceMemory::<f32>::new(n)?;
    
    // Copy input data to device
    d_a.copy_from_host(&a)?;
    d_b.copy_from_host(&b)?;
    
    // Set up grid and block dimensions
    let block_size = 256;
    let grid_dim = calculate_grid_1d(n as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);
    
    println!("Launching kernel with grid={:?}, block={:?}", grid_dim, block_dim);
    
    // Launch the kernel
    unsafe {
        // For this example, we're just going to show the structure
        // In a real application, you'd have an actual kernel to execute
        // This code won't actually run since we're not linking against a real kernel
        launch_vector_add(
            grid_dim,
            block_dim,
            0, // shared memory bytes
            Some(&stream),
            d_a.as_ptr() as *const f32,
            d_b.as_ptr() as *const f32,
            d_c.as_ptr() as *mut f32,
            n as u32
        )?;
    }
    
    // Synchronize
    stream.synchronize()?;
    
    // Copy results back to host
    d_c.copy_to_host(&mut c)?;
    
    // Verify the first few results
    println!("First 10 results:");
    for i in 0..10.min(n) {
        println!("{} + {} = {}", a[i], b[i], c[i]);
    }
    
    println!("All operations completed!");
    Ok(())
}

// In a real application, the actual kernel would be in a .hip file like this:
/*
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
*/