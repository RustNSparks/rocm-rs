use rocm_rs::rocblas::scal;
use rocm_rs::{hip::*, rocblas};
use std::error::Error;

// this example shows Matrix-Vector Multiply with rocblas
fn main() -> std::result::Result<(), Box<dyn Error>> {
    // Initialize rocBLAS handle
    let handle = rocblas::Handle::new()?;

    // Matrix dimensions
    let m = 2; // rows
    let n = 3; // columns

    // Host data (column-major order)
    let mut h_a: [f32; 6] = [
        1.0, 4.0, // Column 0
        2.0, 5.0, // Column 1
        3.0, 6.0, // Column 2
    ];

    // Device memory pointers
    let mut d_a = DeviceMemory::<f32>::new(m * n)?;

    d_a.copy_from_host(&h_a)?;


    let alpha: f32 = 2.0;
    // Perform y = alpha * A
    scal(&handle, (n*m) as i32, &alpha, &d_a, 1)?;
    

    // Copy result back to host
    d_a.copy_to_host(&mut h_a)?;

    println!("Result: {:?}", h_a);

    Ok(())
}
