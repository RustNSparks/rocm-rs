# Vector Addition Example

This example demonstrates how to use the rocm-rs library to perform a simple vector addition operation on an AMD GPU using HIP.

## Components

- `kernel.hip`: Contains the HIP kernel for vector addition
- `build.sh`: Script to compile the kernel to a binary file
- `main.rs`: Rust application that loads the kernel and executes it

## Building and Running

### Step 1: Compile the HIP kernel

First, make sure you have ROCm installed and configured properly on your system.

Then run the build script to compile the kernel:

```bash
chmod +x build.sh
./build.sh
```

This will create a file named `vector_add.hsaco` which contains the compiled kernel.

### Step 2: Build and run the Rust application

```bash
cargo build --release
cp vector_add.hsaco target/release/
cargo run --release
```

You can specify the vector size as a command-line argument:

```bash
cargo run --release -- 10000000
```

## What this example demonstrates

1. **Loading Modules**: How to load a precompiled HIP kernel
2. **Memory Management**: Allocating and copying memory between host and device
3. **Kernel Execution**: Setting up and launching a kernel with proper parameters
4. **Performance Measurement**: Using the Timer API to measure performance of different operations
5. **Error Handling**: Using the unified error handling system

## Troubleshooting

- If the kernel file isn't found, make sure you've run the build script and copied the .hsaco file to the same directory as the executable
- Check that your ROCm installation is working correctly
- Verify that you have a compatible AMD GPU

## Expected Output

The example prints timing information for:
- Host-to-device memory transfer
- Kernel execution
- Device-to-host memory transfer

It also verifies the results by comparing them with a CPU computation and prints sample values from the beginning and end of the vector.