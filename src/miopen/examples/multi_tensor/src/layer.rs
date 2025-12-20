use std::rc::Rc;

use rocm_rs::{error::Result, hip::{DeviceMemory, Dim3, Function, Module, kernel::AsKernelArg}, kernel_args, miopen::{self, ActivationDescriptor, ActivationMode, DataType, TensorDescriptor}};


const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

pub struct Layer {
    tensor_desc: TensorDescriptor,
    activation_desc: ActivationDescriptor,
    device_act: DeviceMemory<f32>,
    device_grad_pre: DeviceMemory<f32>,
    pub(crate) device_grad_act: DeviceMemory<f32>,
    grad_pre: Vec<f32>,
    input_grad: Vec<f32>,
    input_size: usize,
    output_size: usize,
    device_weights: DeviceMemory<f32>,
    device_bias: DeviceMemory<f32>,
    device_output: DeviceMemory<f32>,
    _module: Rc<Module>,
    function: Function,
}

impl Layer {
    pub fn new(
        output_size: usize,
        input_size: usize,
        activation_mode: ActivationMode,
        module: Rc<Module>,
    ) -> Result<Self> {
        let function = module.get_function("linear_transform")?;
        Ok(Self {
            tensor_desc: TensorDescriptor::new_4d(
                DataType::MiopenFloat,
                1,
                output_size as i32,
                1,
                1,
            )?,
            activation_desc: ActivationDescriptor::with_mode(activation_mode, 0.0, 0.0, 0.0)?,
            device_act: DeviceMemory::new(output_size)?,
            device_grad_pre: DeviceMemory::new(output_size)?,
            device_grad_act: DeviceMemory::new(output_size)?,
            grad_pre: vec![0.0; output_size],
            input_grad: vec![0.0; input_size],
            input_size,
            output_size,
            device_weights: DeviceMemory::new(output_size * input_size)?,
            device_bias: DeviceMemory::new(output_size)?,
            device_output: DeviceMemory::new(output_size)?,
            _module: module,
            function,
        })
    }

    pub fn input_grad(&self) -> &[f32] {
        &self.input_grad
    }

    pub fn forward(
        &mut self,
        handle: &miopen::Handle,
        input: &DeviceMemory<f32>,
        weights: &[f32],
        bias: &[f32],
    ) -> Result<&DeviceMemory<f32>> {
        self.device_weights.copy_from_host(weights)?;
        self.device_bias.copy_from_host(bias)?;

        let args = kernel_args!(
            input,
            self.device_weights,
            self.device_bias,
            self.device_output,
            self.input_size,
            self.output_size
        );

        self.function.launch(
            Dim3::new_1d(self.output_size as u32),
            Dim3::new_1d(1),
            0,
            None,
            args,
        )?;

        self.activation_desc.forward(
            handle,
            &ALPHA,
            &self.tensor_desc,
            &self.device_output,
            &BETA,
            &self.tensor_desc,
            &mut self.device_act,
        )?;

        Ok(&self.device_act)
    }

    pub fn backward(
        &mut self,
        handle: &miopen::Handle,
        prev_activations: &DeviceMemory<f32>,
        weights: &mut [f32],
        bias: &mut [f32],
        learning_rate: f32,
    ) -> Result<()> {
        self.activation_desc.backward(
            handle,
            &ALPHA,
            &self.tensor_desc,
            &self.device_act,
            &self.tensor_desc,
            &self.device_grad_act,
            &self.tensor_desc,
            &self.device_output,
            &BETA,
            &self.tensor_desc,
            &mut self.device_grad_pre,
        )?;
        self.device_grad_pre.copy_to_host(&mut self.grad_pre)?;

        let input_size = self.input_size;

        let prev_activations = {
            let mut vec = vec![0.0; input_size];
            prev_activations.copy_to_host(&mut vec)?;
            vec
        };

        for (i, grad_in) in self.input_grad.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (o, &grad_out) in self.grad_pre.iter().enumerate() {
                sum += weights[o * input_size + i] * grad_out;
            }
            *grad_in = sum;
        }

        for (o, &grad) in self.grad_pre.iter().enumerate() {
            let start = o * self.input_size;
            for input_idx in 0..self.input_size {
                weights[start + input_idx] -= learning_rate * grad * prev_activations[input_idx];
            }
            bias[o] -= learning_rate * grad;
        }

        Ok(())
    }
}
