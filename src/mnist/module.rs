use super::{optimizer::Optimizer, types::Tensor};

/// Core trait for layers and models.
pub trait Module {
    /// Forward pass: input -> output.
    fn forward(&self, input: &Tensor) -> Tensor;
    /// Backward pass: gradient of output -> gradient of input.
    fn backward(&mut self, grad_output: &Tensor) -> crate::mnist::tensor::Tensor;
    /// Update parameters using optimizer.
    fn update(&mut self, optimizer: &mut dyn Optimizer);
}
