/// Core trait for layers and models.
pub trait Module {
    /// Forward pass: input -> output.
    fn forward(&mut self, input: &crate::mnist::tensor::Tensor) -> crate::mnist::tensor::Tensor;
    /// Backward pass: gradient of output -> gradient of input.
    fn backward(&mut self, grad_output: &crate::mnist::tensor::Tensor) -> crate::mnist::tensor::Tensor;
    /// Update parameters using optimizer.
    fn update(&mut self, optimizer: &mut dyn crate::mnist::optim::Optimizer);
}
