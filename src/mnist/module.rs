/// Core trait for layers and models.
pub trait Module {
    type InternalCache;
    type InputData;
    type OutputData;
    /// Forward pass: input -> output.
    fn forward(&self, input: &Self::InputData) -> (Self::OutputData, Self::InternalCache);
    /// Backward pass: gradient of output -> gradient of input.
    fn backward(
        &self,
        grad_output: &Self::OutputData,
        internal_cache: &Self::InternalCache,
    ) -> Self::InputData;
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

// Needs to work with more generic data schemas.
pub trait Optimizer {
    // Update parameters given slices of parameters and corresponding gradients.
    // fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]);
}
//
// impl Optimizer for SGD {
//     fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]) {
//         for (param, grad) in params.iter_mut().zip(grads.iter()) {
//             let p = &mut *param;
//             for (pv, gv) in p.data.iter_mut().zip(grad.data.iter()) {
//                 *pv -= self.lr * gv;
//             }
//         }
//     }
// }
