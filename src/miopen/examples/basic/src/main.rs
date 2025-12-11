use rocm_rs::{
    error::Result,
    hip::DeviceMemory,
    miopen::{
        self, ActivationDescriptor, TensorDescriptor,
        ffi::{miopenActivationMode_t_miopenActivationRELU, miopenDataType_t_miopenFloat},
    },
};

fn main() -> Result<()> {
    // -----------------------
    // 1. MIOpen handle
    // -----------------------
    let miopen = miopen::Handle::new()?;

    // -----------------------
    // 2. Training data
    // -----------------------
    let x_host = vec![0f32, 1., 2., 3.];
    let y_target = vec![0f32, 1., 2., 3.];
    let mut y_pred = vec![0f32; 4];
    let mut dl_dy = vec![0f32; 4];

    // -----------------------
    // 3. Allocate GPU buffers
    // -----------------------
    let mut d_linear = DeviceMemory::<f32>::new(4)?; // stores wx+b
    let d_y = DeviceMemory::<f32>::new(4)?; // ReLU output
    let mut d_dy = DeviceMemory::<f32>::new(4)?; // gradient input
    let d_dx = DeviceMemory::<f32>::new(4)?; // gradient output

    // -----------------------
    // 4. Tensor descriptor
    // -----------------------
    let mut tensor = TensorDescriptor::new()?;
    tensor.set_4d(miopenDataType_t_miopenFloat, 1, 1, 1, 4)?;

    // -----------------------
    // 5. ReLU activation
    // -----------------------
    let mut activation = ActivationDescriptor::new()?;
    activation.set(miopenActivationMode_t_miopenActivationRELU, 0.0, 0.0, 0.0)?;

    let alpha = 1f32.to_ne_bytes();
    let beta = 0f32.to_ne_bytes();

    // -----------------------
    // 6. Parameters of our 1-neuron model
    // -----------------------
    let mut w: f32 = 0.1;
    let mut b: f32 = 0.0;
    let lr: f32 = 0.01;

    // -----------------------
    // 7. Training loop
    // -----------------------
    for epoch in 0..200 {
        // ---- forward linear: wx + b ----
        for i in 0..4 {
            y_pred[i] = w * x_host[i] + b;
        }

        d_linear.copy_from_host(&y_pred)?;

        // ---- MIOpen forward: ReLU(wx+b) ----
        unsafe {
            activation.forward(
                &miopen,
                &alpha,
                &tensor,
                d_linear.as_ptr(),
                &beta,
                &tensor,
                d_y.as_ptr(),
            )?
        }

        // bring prediction back
        d_y.copy_to_host(&mut y_pred)?;

        // ---- compute dL/dy = 2*(y_pred - y_target) ----
        let mut loss = 0.0;
        for i in 0..4 {
            let err = y_pred[i] - y_target[i];
            loss += err * err;
            dl_dy[i] = 2.0 * err;
        }

        d_dy.copy_from_host(&dl_dy)?;

        // ---- MIOpen backward: dL/dx = ReLU'(x)*dL/dy ----
        unsafe {
            activation.backward(
                &miopen,
                &alpha,
                &tensor,
                d_y.as_ptr(), // y from forward
                &tensor,
                d_dy.as_ptr(), // dL/dy
                &tensor,
                d_linear.as_ptr(), // x before activation
                &beta,
                &tensor,
                d_dx.as_ptr(), // output: dL/dx
            )?
        }

        // get dL/dx back
        let mut dl_dx = vec![0f32; 4];
        d_dx.copy_to_host(&mut dl_dx)?;

        // ---- compute gradients for w and b ----
        let grad_w: f32 = dl_dx.iter().zip(x_host.iter()).map(|(dx, x)| dx * x).sum();

        let grad_b: f32 = dl_dx.iter().sum();

        // ---- gradient descent ----
        w -= lr * grad_w;
        b -= lr * grad_b;

        if epoch % 20 == 0 {
            println!(
                "epoch {:3}  loss={:.4}  w={:.3}  b={:.3}",
                epoch, loss, w, b
            );
        }
    }

    println!("\n=== INFERENCE PHASE ===");

    let test_inputs = vec![-2.0f32, -1.0, 0.5, 5.0];
    let mut test_linear = vec![0f32; test_inputs.len()];
    let mut test_output = vec![0f32; test_inputs.len()];

    
    // -----------------------
    // 8. Linear forward
    // -----------------------
    for i in 0..test_inputs.len() {
        test_linear[i] = w * test_inputs[i] + b;
    }

    // -----------------------
    // 9. Copy to GPU
    // -----------------------
    d_linear.copy_from_host(&test_linear)?;

    // -----------------------
    // 10. Inference
    // -----------------------
    unsafe {
        activation.forward(
            &miopen,
            &alpha,
            &tensor,
            d_linear.as_ptr(),
            &beta,
            &tensor,
            d_y.as_ptr(),
        )?
    }

    d_y.copy_to_host(&mut test_output)?;

    // -----------------------
    // 11. Results
    // -----------------------
    for (x, y) in test_inputs.iter().zip(test_output.iter()) {
        println!("input = {:>5}   output = {}", x, y);
    }

    Ok(())
}
