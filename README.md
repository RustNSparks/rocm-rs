# rocm-rs: Safe Rust wrappers for AMD ROCm Libraries

This project provides Rust bindings for AMD's ROCm (Radeon Open Compute) libraries, allowing Rust developers to leverage AMD GPUs for high-performance computing.

## Current Status

**Note: This project is in early development.**

Currently implemented:
- ✅ rocFFT - Fast Fourier Transform library (raw bindings + safe wrappers)
- ✅ HIP - Heterogeneous-Compute Interface for Portability (raw bindings + safe wrappers)
- ✅ rocBLAS - Basic Linear Algebra Subprograms (raw bindings + safe wrappers)
- ✅ MIOpen - Deep learning primitives (raw bindings + safe wrappers)
- ✅ rocRAND - Random number generation (raw bindings + safe wrappers)
- ✅ rocSOLVER - Linear system solvers (raw bindings only)
- ✅ rocSPARSE - Sparse linear algebra (raw bindings only)
- ✅ ROCArray - GPU array struct with api similar to Vec
- ✅ rocmsmi - system managment interface (refer to [rocm_smi_lib](https://github.com/PTFOPlayer/rocm_smi_lib_rs))

The project currently focuses on providing raw FFI bindings for most libraries, with safe Rust wrappers available for rocFFT. Additional safe wrappers for other libraries are planned for future development.

## Prerequisites

- AMD ROCm installed (version 6.3 or later recommended.It may work on older versions, but I did not test that)
- Ubuntu 24.04 (it works on WSL too)
- Rust toolchain (1.65.0 or later recommended)
- A compatible AMD GPU

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rocm-rs = "0.1.0"
```

## Usage

First, ensure that the ROCm libraries are in your library path or set the `ROCM_PATH` environment variable.

### Writing your own kernels with rust

```rust
use std::path::PathBuf;

use rocm_kernel_macros::{amdgpu_kernel_attr, amdgpu_kernel_finalize, amdgpu_kernel_init};
use rocm_rs::hip::*;

const LEN: usize = 1024;

// initializing rust gpu kernel
amdgpu_kernel_init!();

// marking code that will be coppied to gpu kernel
#[amdgpu_kernel_attr]
fn kernel(input: *const u32, output: *mut u32) {
    // retriving data from buffere by workitem
    let mut num = read_by_workitem_id_x(input);
    
    // writing data back
    write_by_workitem_id_x(output, num * 3);
}

// compiling gpu kernel
const AMDGPU_KERNEL_BINARY_PATH: &str = amdgpu_kernel_finalize!();

fn main() -> Result<()> {
    // setting up device
    let device = Device::new(0)?;
    device.set_current()?;

    // loading gpu kerenel (runs in runtime!)
    let kernel_path = PathBuf::from(AMDGPU_KERNEL_BINARY_PATH);
    assert!(kernel_path.exists());

    let module = Module::load(kernel_path)?;

    // acquiring function handle from gpu kernel 
    let function = unsafe { module.get_function("kernel")? };

    // preparing host side buffers
    let mut in_host: Vec<u32> = vec![0; LEN];
    let mut out_host: Vec<u32> = vec![0; LEN];

    for i in 0..LEN {
        in_host[i] = i as u32;
    }

    // preparing gpu side buffers
    let mut input = DeviceMemory::<u32>::new(LEN)?;
    let output = DeviceMemory::<u32>::new(LEN)?;

    input.copy_from_host(&in_host)?;

    // providing arguments for kernel
    let kernel_args = [input.as_kernel_arg(), output.as_kernel_arg()];

    // setting up launch args
    let grid_dim = Dim3 { x: 2, y: 1, z: 1 };
    let block_dim = Dim3 {
        x: (LEN / 2) as u32,
        y: 1,
        z: 1,
    };

    function.launch(grid_dim, block_dim, 0, None, &mut kernel_args.clone())?;

    // retriving computed data
    output.copy_to_host(&mut out_host)?;

    println!("Output: {:?}", &out_host[..256]);

    Ok(())
}

```

### Using rocFFT with safe wrappers:

```rust
use rocm_rs::rocfft::{self, plan, execution, field};

fn main() {
    // Initialize the rocFFT library
    // Use the safe wrappers for rocFFT
    let plan = plan::Plan::new(/* parameters */);
    let field = field::Field::new(/* parameters */);
    let execution = execution::Execution::new(/* parameters */);
    
    // Perform FFT operations
    // ...
}
```

### Using other libraries with raw bindings:

```rust
use rocm_rs::hip::*;

fn main() {
    unsafe {
        // Example of using HIP raw bindings
        let mut device_count = 0;
        hipGetDeviceCount(&mut device_count);
        println!("Found {} HIP devices", device_count);
        
        // Use other raw bindings as needed
        // ...
    }
}
```

## Building from Source

**Important**: When building from source, you need to run `cargo build` first to generate the bindings files before you can use the library or run tests.

```bash
# Clone the repository
git clone https://github.com/radudiaconu0/rocm-rs.git
cd rocm-rs

# Set the ROCm path if not in the default location
export ROCM_PATH=/opt/rocm

# Build the project (generates bindings)
cargo build
```

## Feature flags

- rocm_smi - enables bindings and wrappers for rocm_smi_lib

## Examples
- hip
  - vector_add - example containing kernel written in cpp launched with rocm-rs
  - rust_kernel - example containing kernel writtein in rust using macros 
- rand
  - normal - generating random numbers with normal distribution 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing:
1. Run `cargo build` first to generate the bindings
2. Add tests for new functionality
3. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AMD for developing and maintaining ROCm
- The Rust community for bindgen and other tools used in this project
