use crate::mnist::module::Module;
use crate::mnist::tensor::Tensor;
use crate::mnist::types::Optimizer;

/// Rectified Linear Unit activation.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for x in output.data.iter_mut() {
            *x = x.max(0.0);
        }
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // TODO: use cached input, but using approximate gradient
        let grad_input = grad_output.clone();
        // In real impl, multiply by mask
        grad_input
    }

    fn update(&mut self, _optimizer: &mut dyn Optimizer) {
        // no parameters
    }
}
