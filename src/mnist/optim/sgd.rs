use crate::mnist::tensor::Tensor;
use crate::mnist::optim::Optimizer;

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