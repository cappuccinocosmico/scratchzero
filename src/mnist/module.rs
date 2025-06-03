/// Core trait for layers and models.
pub trait Module {
    type InternalCache;
    /// Forward pass: input -> output.
    fn forward(&self, input: &Tensor) -> (Tensor, Self::InternalCache);
    /// Backward pass: gradient of output -> gradient of input.
    fn backward(&self, grad_output: &Tensor, internal_cache: &Self::InternalCache) -> Tensor;
    /// Update parameters using optimizer.
    fn update(&mut self, optimizer: &mut dyn Optimizer);
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let p = &mut *param;
            for (pv, gv) in p.data.iter_mut().zip(grad.data.iter()) {
                *pv -= self.lr * gv;
            }
        }
    }
}

pub trait Optimizer {
    /// Update parameters given slices of parameters and corresponding gradients.
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]);
}
