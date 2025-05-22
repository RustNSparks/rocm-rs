use rocm_rs::error::Result;
use rocm_rs::hip::{
    Device, DeviceMemory, Stream, Module, Function,
    calculate_grid_1d, Dim3, device_synchronize, Timer
};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize device
    println!("Initializing device...");
    let device = Device::new(0)?;
    device.set_current()?;
    
    // Print device info
    let props = device.properties()?;
    println!("Using device: {}", props.name);
    println!("Compute capability: {}.{}", props.major, props.minor);
    println!("Multiprocessor count: {}", props.multi_processor_count);
    
    // Load the precompiled kernel module
    let kernel_path = PathBuf::from("vector_add.hsaco");

    if !kernel_path.exists() {
        println!("Error: Could not find kernel file: {}", kernel_path.display());
        println!(
            "Error: Could not find kernel file: {}",
            kernel_path.display()
        );
        println!("Make sure to run the build.sh script first to compile the kernel.");
        return Ok(());
    }
    
    println!("Loading kernel module from: {}", kernel_path.display());
    let module = Module::load(kernel_path)?;
    
    // Get the function handle
    let function = unsafe { module.get_function("vector_add")? };
    
    // Create a stream for async operations
    let stream = Stream::new()?;
    
    // Get the test size from command line or use default
    let args: Vec<String> = env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(1_000_000)
    } else {
        1_000_000
    };
    
    println!("Vector size: {}", n);
    
    // Prepare host data
    println!("Preparing host data...");
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (2.0 * i as f32)).collect();
    let mut c = vec![0.0f32; n];
    
    // Allocate device memory
    println!("Allocating device memory...");
    let d_a = DeviceMemory::<f32>::new(n)?;
    let d_b = DeviceMemory::<f32>::new(n)?;
    let d_c = DeviceMemory::<f32>::new(n)?;
    
    // Create a timer
    println!("Creating timer...");
    let timer = Timer::new()?;
    
    // Start timing host-to-device transfer
    timer.start(&stream)?;
    
    // Copy data from host to device
    println!("Copying host data to device...");
    d_a.copy_from_host_async(a.clone(), &stream)?;
    d_b.copy_from_host_async(b.clone(), &stream)?;
    
    // Stop timing host-to-device transfer
    timer.stop(&stream)?;
    let h2d_time = timer.elapsed_time()?;
    
    // Set up kernel launch parameters
    let block_size = 256;
    let grid_dim = calculate_grid_1d(n as u32, block_size);
    let block_dim = Dim3::new_1d(block_size);

    println!(
        "Launching kernel with grid={}, block={}",
        grid_dim.x, block_dim.x
    );
    
    // Prepare kernel arguments
    let n_u32 = n as u32;
    let kernel_args = [
        d_a.as_kernel_arg(),
        d_b.as_kernel_arg(),
        d_c.as_kernel_arg(),
        &n_u32 as *const _ as *mut std::ffi::c_void,
    ];
    
    // Start timing kernel execution
    timer.start(&stream)?;
    
    // Launch the kernel
    function.launch(
        grid_dim,
        block_dim,
        0, // shared memory bytes
        Some(&stream),
        &mut kernel_args.clone(),
    )?;
    
    // Stop timing kernel execution
    timer.stop(&stream)?;
    let kernel_time = timer.elapsed_time()?;
    
    // Start timing device-to-host transfer
    timer.start(&stream)?;
    
    // Copy results back to host
    let pending = d_c.copy_to_host_async(c, &stream)?;

    // Synchronize the stream to ensure all operations are complete
    let c = stream.synchronize_memory(pending)?;
    
    // Stop timing device-to-host transfer
    timer.stop(&stream)?;
    let d2h_time = timer.elapsed_time()?;
    
    // Print timing information
    println!("Host to Device Transfer: {:.3} ms", h2d_time);
    println!("Kernel Execution: {:.3} ms", kernel_time);
    println!("Device to Host Transfer: {:.3} ms", d2h_time);
    println!("Total GPU Time: {:.3} ms", h2d_time + kernel_time + d2h_time);
    
    // Verify results
    println!("Verifying results...");
    let cpu_start = Instant::now();
    
    let mut all_correct = true;
    for i in 0..n {
        let expected = a[i] + b[i];
        let actual = c[i];
        if (expected - actual).abs() > 1e-5 {
            println!(
                "Error at index {}: expected {}, got {}",
                i, expected, actual
            );
            all_correct = false;
            if i > 10 {
                println!("Stopping verification after 10 errors...");
                break;
            }
        }
    }
    
    let cpu_elapsed = cpu_start.elapsed();
    println!("CPU verification time: {:.3} ms", cpu_elapsed.as_secs_f32() * 1000.0);
    
    if all_correct {
        println!("All results are correct!");
    } else {
        println!("Some errors were found in the results.");
    }
    
    // Print a few results
    if n > 5 {
        println!("First 5 results:");
        for i in 0..5 {
            println!("c[{}] = {} + {} = {}", i, a[i], b[i], c[i]);
        }
        
        println!("Last 5 results:");
        for i in n-5..n {
            println!("c[{}] = {} + {} = {}", i, a[i], b[i], c[i]);
        }
    }
    
    println!("Example completed successfully!");
    Ok(())
}