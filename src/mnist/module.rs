/// Core trait for layers and models.
pub trait Module {
    type InternalData;
    type CacheData;
    type InputData;
    type OutputData;
    /// Forward pass with cache: input -> output, cache.
    fn forward_with_cache(&self, input: &Self::InputData) -> (Self::OutputData, Self::CacheData);
    /// Forward pass without cache.
    fn forward(&self, input: &Self::InputData) -> Self::OutputData {
        self.forward_with_cache(input).0
    }
    /// Backward pass: gradient of output -> gradient of input, gradient of layer weights
    fn backward(
        &self,
        grad_output: &Self::OutputData,
        cache: &Self::CacheData,
    ) -> (Self::InputData, Self::InternalData);
    /// Apply an offset to the weights.
    fn apply_offset_to_weights(&mut self, offset: &Self::InternalData);
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
