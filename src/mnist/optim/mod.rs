pub trait Optimizer {
    /// Update parameters given slices of parameters and corresponding gradients.
    fn step(&mut self, params: &mut [&mut crate::mnist::tensor::Tensor], grads: &[&crate::mnist::tensor::Tensor]);
}

pub mod sgd;
