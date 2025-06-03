use crate::mnist::module::Module;
use crate::mnist::optim::Optimizer;
use crate::mnist::tensor::Tensor;

/// Container to apply layers sequentially.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in self.layers.iter_mut() {
            x = layer.forward(&x);
        }
        x
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    fn update(&mut self, optimizer: &mut dyn Optimizer) {
        for layer in self.layers.iter_mut() {
            layer.update(optimizer);
        }
    }
}