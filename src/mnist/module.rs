use crate::mnist::field::Field;
use crate::mnist::tensor::Tensor;

pub trait ModConfig {
    /// The type of input this module consumes.
    type Input;
    /// The type of output this module produces.
    type Output;
    /// The type of this module (pure values), and also the type of the gradient.
    type Param;
    /// Any cached activations needed for backward.
    type Cache;
}
/// Each layer or model that has (optional) parameters implements `Module`.
pub trait Module: ModConfig {
    /// Pure forward: returns output and cache needed for backward.
    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache);

    /// Pure backward: given grad w.r.t. output + cache, returns
    /// - grad w.r.t. input, and
    /// - grad w.r.t. this module's parameters.
    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::Param);

    /// Apply an optimizer update to the parameters, given parameter gradients and learning rate.
    fn update(&mut self, param_grad: &Self::Param, lr: f32);
}
