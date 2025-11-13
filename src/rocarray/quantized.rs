// src/rocarray/quantized_stub.rs
// TEAM-506: ROCm quantized kernels (runtime-compiled from quantized.hip)
// 
// Converted from CUDA using hipify-perl + manual ROCm-specific fixes
// Source: candle-kernels/src/quantized.cu (via hipify-perl)
// 
// Status: ✅ WORKS WITHOUT PRE-COMPILATION (uses runtime compilation like CUDA)

/// HIP source code for quantized kernels
/// 
/// This follows the SAME pattern as CUDA (candle-kernels):
/// - CUDA: Embeds PTX text, compiles at runtime
/// - ROCm: Embeds HIP source, compiles at runtime via hipcc
/// 
/// Contains 103 kernels:
/// - Dequantization kernels (F32 + F16): 22 kernels
/// - Quantization kernels: 1 kernel (quantize_q8_1)
/// - Fused dequant+matmul kernels: 10 kernels
/// - Batch matmul kernels (sizes 1-8): 60 kernels
/// - Full matmul kernels: 10 kernels
/// 
/// ## How It Works (Same as CUDA)
/// 
/// 1. **CUDA approach:**
///    ```rust
///    pub const QUANTIZED: &str = include_str!("quantized.ptx");  // PTX text
///    // Driver compiles PTX → CUBIN at runtime
///    ```
/// 
/// 2. **ROCm approach (THIS FILE):**
///    ```rust
///    pub const QUANTIZED: &str = include_str!("quantized.hip");  // HIP source
///    // hipcc compiles HIP → HSACO at runtime
///    ```
/// 
/// ## No Pre-Compilation Required!
/// 
/// Just like CUDA, the kernels are compiled the first time they're used.
/// This requires `hipcc` to be installed, but that's already a requirement
/// for ROCm development (same as `nvcc` for CUDA).
/// 
/// ## Wiring
/// 
/// This constant is used by Candle's ROCm backend:
/// - `candle-core/src/quantized/rocm.rs` loads kernels via `dev.get_or_load_func()`
/// - Runtime compilation happens via `hip::compile_and_load()`
/// - No additional Rust wrappers needed - kernels are loaded dynamically
/// 
/// ## Kernel Loading Example
/// 
/// ```rust
/// // From candle-core/src/quantized/rocm.rs:63
/// let func = dev.get_or_load_func("quantize_q8_1", quantized_stub::QUANTIZED)?;
/// func.launch(grid_dim, block_dim, 0, None, &mut kernel_params)?;
/// ```

/// HIP source code for quantized kernels (runtime-compiled like CUDA PTX)
pub const QUANTIZED: &str = include_str!("quantized.hip");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_source_exists() {
        // Verify the HIP source code is embedded
        assert!(QUANTIZED.len() > 1000, "HIP source should be at least 1KB");
        assert!(QUANTIZED.contains("quantize_q8_1"), "Should contain quantize_q8_1 kernel");
    }

    #[test]
    #[ignore = "Requires hipcc installed - runtime compilation test"]
    fn quantized_kernels_compilable() {
        // This test verifies that the HIP source can be compiled at runtime
        // Requires hipcc to be installed (same as CUDA requires nvcc)
        use crate::hip;
        
        let module = hip::compile_and_load(QUANTIZED, &[]).expect("Failed to compile HIP kernels");
        let func = module.get_function("quantize_q8_1").expect("Failed to get quantize_q8_1 function");
        
        // If we got here, compilation succeeded!
        drop(func);
        drop(module);
    }
}
