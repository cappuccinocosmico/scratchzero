use crate::mnist::tensor::Tensor;
use crate::mnist::module::Module;
use crate::mnist::optim::Optimizer;

/// Rectified Linear Unit activation.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for x in output.data.iter_mut() {
            *x = x.max(0.0);
        }
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // TODO: use cached input, but using approximate gradient
        let mut grad_input = grad_output.clone();
        // In real impl, multiply by mask
        grad_input
    }

    fn update(&mut self, _optimizer: &mut dyn Optimizer) {
        // no parameters
    }
}

/// Flatten layer: flatten input to 2D [batch, features]
pub struct Flatten {
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten { input_shape: None }
    }
}

impl Module for Flatten {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let batch = input.shape[0];
        let features: usize = input.shape.iter().skip(1).product();
        self.input_shape = Some(input.shape.clone());
        let mut output = input.clone();
        output.reshape(&[batch, features]);
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();
        if let Some(shape) = &self.input_shape {
            grad.reshape(shape);
        }
        grad
    }

    fn update(&mut self, _optimizer: &mut dyn Optimizer) {
        // no parameters
    }
}
