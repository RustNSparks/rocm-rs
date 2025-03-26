# rocm-rs: Rust Bindings for AMD ROCm Libraries

This project provides Rust bindings for AMD's ROCm (Radeon Open Compute) libraries, allowing Rust developers to leverage AMD GPUs for high-performance computing.

## Current Status

**Note: This project is in early development.**

Currently implemented:
- ✅ rocFFT - Fast Fourier Transform library (raw bindings + safe wrappers)
- ✅ HIP - Heterogeneous-Compute Interface for Portability (raw bindings only)
- ✅ rocBLAS - Basic Linear Algebra Subprograms (raw bindings only)
- ✅ MIOpen - Deep learning primitives (raw bindings only)
- ✅ rocRAND - Random number generation (raw bindings + safe wrappers)
- ✅ rocSOLVER - Linear system solvers (raw bindings only)
- ✅ rocSPARSE - Sparse linear algebra (raw bindings only)

The project currently focuses on providing raw FFI bindings for most libraries, with safe Rust wrappers available for rocFFT. Additional safe wrappers for other libraries are planned for future development.

## Prerequisites

- AMD ROCm installed (version 6.0 or later recommended)
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

# Run tests
cargo test
```

## Project Structure

```
rocm-rs/
├── Cargo.toml
├── build.rs              # Build script to generate bindings
├── include/              # C/C++ headers
│   ├── hip.h
│   ├── miopen.h
│   ├── rocblas.h  
│   ├── rocfft.h
│   ├── rocrand.h
│   ├── rocsolver.h
│   └── rocsparse.h
├── src/
│   ├── lib.rs            # Main library entry point
│   ├── hip/              # HIP module
│   │   ├── mod.rs
│   │   └── bindings.rs   # Auto-generated bindings (do not edit)
│   ├── miopen/           # MIOpen module
│   │   ├── mod.rs
│   │   └── bindings.rs
│   ├── rocblas/          # rocBLAS module
│   │   ├── mod.rs
│   │   └── bindings.rs
│   ├── rocfft/           # rocFFT module (with safe wrappers)
│   │   ├── mod.rs
│   │   ├── bindings.rs
│   │   ├── cache.rs
│   │   ├── description.rs
│   │   ├── error.rs
│   │   ├── execution.rs
│   │   ├── fft.rs
│   │   ├── field.rs
│   │   └── plan.rs
│   ├── rocrand/          # rocRAND module
│   │   ├── mod.rs
│   │   └── bindings.rs
│   ├── rocsolver/        # rocSOLVER module
│   │   ├── mod.rs
│   │   └── bindings.rs
│   └── rocsparse/        # rocSPARSE module
│       ├── mod.rs
│       └── bindings.rs
```

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
