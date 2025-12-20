mod data;
mod kernels;
pub mod layer;

use crate::{kernels::KERNEL, layer::Layer};
use rocm_rs::{
    error::Result,
    hip::{DeviceMemory, Dim3, Module, kernel::AsKernelArg},
    kernel_args,
    miopen::{self, ActivationMode},
};
use std::rc::Rc;

use crate::data::prepare_data;

const HIDDEN_SIZE: usize = 8;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 200;

fn main() -> Result<()> {
    let (x_data, y_target, class_labels) = prepare_data();

    let handle = miopen::Handle::new()?;

    let input_size = x_data[0].len();
    let hidden_size = HIDDEN_SIZE;
    let output_size = class_labels.len();
    let learning_rate = LEARNING_RATE;
    let epochs = EPOCHS;

    let module = Rc::new(Module::load_data(KERNEL)?);

    let mut hidden_layer = Layer::new(
        hidden_size,
        input_size,
        ActivationMode::MiopenActivationLOGISTIC,
        module.clone(),
    )?;

    let mut output_layer = Layer::new(
        output_size,
        hidden_size,
        ActivationMode::MiopenActivationSOFTRELU,
        module.clone(),
    )?;

    let mut weights_input_hidden = init_weights(hidden_size, input_size);
    let mut bias_hidden = vec![0.0; hidden_size];
    let mut weights_hidden_output = init_weights(output_size, hidden_size);
    let mut bias_output = vec![0.0; output_size];

    let gradient_func = module.get_function("gradient")?;
    let mut target_device = DeviceMemory::new(output_size)?;
    let mut x_sample_dev = DeviceMemory::new(input_size)?;

    for epoch in 0..epochs {
        for (x_sample, target) in x_data.iter().zip(y_target.iter()) {
            target_device.copy_from_host(target)?;
            x_sample_dev.copy_from_host(x_sample)?;

            let hidden_activation = hidden_layer.forward(
                &handle,
                &x_sample_dev,
                &weights_input_hidden,
                &bias_hidden,
            )?;

            let prediction = output_layer.forward(
                &handle,
                &hidden_activation,
                &weights_hidden_output,
                &bias_output,
            )?;

            gradient_func.launch(
                Dim3::new_1d(output_size as u32),
                Dim3::new_1d(1),
                0,
                None,
                kernel_args!(
                    prediction,
                    &target_device,
                    &output_layer.device_grad_act,
                    output_size
                ),
            )?;

            output_layer.backward(
                &handle,
                &hidden_activation,
                &mut weights_hidden_output,
                &mut bias_output,
                learning_rate,
            )?;

            hidden_layer
                .device_grad_act
                .copy_from_host(output_layer.input_grad())?;

            hidden_layer.backward(
                &handle,
                &x_sample_dev,
                &mut weights_input_hidden,
                &mut bias_hidden,
                learning_rate,
            )?;
        }

        if epoch % 10 == 0 {
            println!("Epoch {epoch}");
        }
    }

    println!("Inference after training:");

    let inference_samples = vec![
        (vec![5.1, 3.5, 1.4, 0.2], "setosa"),
        (vec![7.0, 3.2, 4.7, 1.4], "versicolor"),
        (vec![6.0, 2.2, 5.0, 1.5], "virginica"),
    ];

    for (features, expected_label) in inference_samples {
        let mut features_dev = DeviceMemory::new(features.len())?;
        features_dev.copy_from_host(&features)?;

        let hidden_activation =
            hidden_layer.forward(&handle, &features_dev, &weights_input_hidden, &bias_hidden)?;

        let prediction = output_layer.forward(
            &handle,
            &hidden_activation,
            &weights_hidden_output,
            &bias_output,
        )?;

        let prediction = {
            let mut vec = vec![0.0; output_size];
            prediction.copy_to_host(&mut vec)?;
            vec
        };

        let predicted_idx = prediction
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap();

        println!(
            "Expected: {expected_label}, Predicted: {}, Probabilities: {:?}",
            class_labels[predicted_idx], prediction
        );
    }

    Ok(())
}

/// Scaling factor used to generate deterministic but non-uniform initial
/// weights. The value 0.37 is approximately 1/e and is chosen to spread
/// the input to `sin` across different phases while keeping the magnitude
/// below 1.0. This constant can be adjusted to change the initialization
/// without altering the overall scheme.
const WEIGHT_INIT_SEED_SCALE: f32 = 0.37;

fn init_weights(rows: usize, cols: usize) -> Vec<f32> {
    let mut weights = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let seed = (row * cols + col) as f32 * WEIGHT_INIT_SEED_SCALE;
            weights.push(seed.sin() * 0.1);
        }
    }
    weights
}
