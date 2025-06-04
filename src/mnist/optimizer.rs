use super::types::Tensor;

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

pub trait Optimizer {
    /// Update parameters given slices of parameters and corresponding gradients.
    fn step<const dim: usize>(&mut self, params: &mut [&mut Tensor<dim>], grads: &[&Tensor<dim>]);
}
impl Optimizer for SGD {}
