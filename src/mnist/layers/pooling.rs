use crate::mnist::tensor::Tensor;
use crate::mnist::module::Module;
use crate::mnist::optim::Optimizer;

/// 2D Max pooling layer.
pub struct MaxPool2D {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    input_cache: Option<Tensor>,
}

impl MaxPool2D {
    pub fn new(kh: usize, kw: usize, sh: usize, sw: usize) -> Self {
        MaxPool2D { kernel: (kh, kw), stride: (sh, sw), input_cache: None }
    }
}

impl Module for MaxPool2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // TODO: implement max pooling
        self.input_cache = Some(input.clone());
        // placeholder: return input clone
        input.clone()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // TODO: implement backward
        let input = self.input_cache.take().expect("No input cached");
        Tensor::zeros(&input.shape)
    }

    fn update(&mut self, _optimizer: &mut dyn Optimizer) {
        // no parameters
    }
}