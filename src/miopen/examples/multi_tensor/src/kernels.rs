use rocm_rs::rocm_kernel_macros::{amdgpu_global, amdgpu_kernel_finalize, amdgpu_kernel_init};

amdgpu_kernel_init!();

#[amdgpu_global]
fn linear_transform(
    input: *const f32,
    weights: *const f32,
    bias: *const f32,
    output: *mut f32,
    input_size: usize,
    output_size: usize,
) {
    let idx = workgroup_id_x() as usize;

    if idx < output_size {
        unsafe {
            let mut sum = *bias.add(idx);
            let offset = idx * input_size;
            for i in 0..input_size {
                sum += *weights.add(offset + i) * *input.add(i);
            }

            *output.add(idx) = sum;
        }
    }
}

#[amdgpu_global]
fn gradient(predicted: *const f32, target: *const f32, grad_output: *mut f32, size: usize) {
    let idx = workgroup_id_x() as usize;

    if idx < size {
        unsafe {
            *grad_output.add(idx) = *predicted.add(idx) - *target.add(idx);
        }
    }
}

pub const KERNEL: &[u8] = include_bytes!(amdgpu_kernel_finalize!());
