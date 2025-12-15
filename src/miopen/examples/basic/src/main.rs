use rocm_rs::{
    error::Result,
    hip::DeviceMemory,
    miopen::{self, ActivationDescriptor, ActivationMode, DataType, TensorDescriptor},
};

fn main() -> Result<()> {
    // -----------------------
    // 1. MIOpen handle
    // -----------------------
    let miopen = miopen::Handle::new()?;

    // -----------------------
    // 2. Training data
    // -----------------------
    let input_len = 8usize;

    let x_host = vec![-10., -2., -1., 0., 1., 2., 3., 10.];
    let y_target = vec![0.0, 0., 0., 0.5, 1., 1., 1., 1.];
    let mut y_pred = vec![0f32; input_len];
    let mut dl_dy = vec![0f32; input_len];

    // -----------------------
    // 3. Allocate GPU buffers
    // -----------------------
    let mut d_linear = DeviceMemory::<f32>::new(input_len)?; // stores wx+b
    let mut d_y = DeviceMemory::<f32>::new(input_len)?; // ReLU output
    let mut d_dy = DeviceMemory::<f32>::new(input_len)?; // gradient input
    let mut d_dx = DeviceMemory::<f32>::new(input_len)?; // gradient output

    // -----------------------
    // 4. Tensor descriptor
    // -----------------------
    let tensor = TensorDescriptor::new_4d(DataType::MiopenFloat, 1, 1, 1, input_len as i32)?;

    // -----------------------
    // 5. ReLU activation
    // -----------------------
    let activation =
        ActivationDescriptor::with_mode(ActivationMode::MiopenActivationLOGISTIC, 0.0, 0.0, 0.0)?;

    let alpha = 1f32;
    let beta = 0f32;

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
        for i in 0..input_len {
            y_pred[i] = w * x_host[i] + b;
        }

        d_linear.copy_from_host(&y_pred)?;

        // ---- MIOpen forward: ReLU(wx+b) ----

        activation.forward(
            &miopen, &alpha, &tensor, &d_linear, &beta, &tensor, &mut d_y,
        )?;

        // bring prediction back
        d_y.copy_to_host(&mut y_pred)?;

        // ---- compute dL/dy = 2*(y_pred - y_target) ----
        let mut loss = 0.0;
        for i in 0..input_len {
            let err = y_pred[i] - y_target[i];
            loss += err * err;
            dl_dy[i] = 2.0 * err;
        }

        d_dy.copy_from_host(&dl_dy)?;

        // ---- MIOpen backward: dL/dx = ReLU'(x)*dL/dy ----
        unsafe {
            activation.backward(
                &miopen, &alpha, &tensor, &d_y, // y from forward
                &tensor, &d_dy, // dL/dy
                &tensor, &d_linear, // x before activation
                &beta, &tensor, &mut d_dx, // output: dL/dx
            )?
        }

        // get dL/dx back
        let mut dl_dx = vec![0f32; input_len];
        d_dx.copy_to_host(&mut dl_dx)?;

        // ---- compute gradients for w and b ----
        let grad_w: f32 = dl_dx.iter().zip(x_host.iter()).map(|(dx, x)| dx * x).sum();

        let grad_b: f32 = dl_dx.iter().sum();

        // ---- gradient descent ----
        w -= lr * grad_w;
        b -= lr * grad_b;

        if epoch % 10 == 0 {
            println!(
                "epoch {:3}  loss={:.4}  w={:.3}  b={:.3}",
                epoch, loss, w, b
            );
        }
    }

    println!("\n=== INFERENCE PHASE ===");

    let test_inputs = vec![-2.0, -1.0, 0., 1.0, 5.0];
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
    activation.forward(
        &miopen, &alpha, &tensor, &d_linear, &beta, &tensor, &mut d_y,
    )?;

    d_y.copy_to_host(&mut test_output)?;

    // -----------------------
    // 11. Results
    // -----------------------
    for (x, y) in test_inputs.iter().zip(test_output.iter()) {
        println!("input = {:>5}   output = {}", x, y);
    }

    Ok(())
}
