use crate::mnist::tensor::Tensor;
use crate::mnist::module::Module;
use crate::mnist::optim::Optimizer;

/// Fully connected layer.
pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Tensor,    // [out, in]
    pub bias: Tensor,      // [out]
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
    input_cache: Option<Tensor>,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w_shape = &[out_features, in_features];
        let b_shape = &[out_features];
        Dense {
            in_features,
            out_features,
            weight: Tensor::zeros(w_shape),
            bias: Tensor::zeros(b_shape),
            grad_weight: Tensor::zeros(w_shape),
            grad_bias: Tensor::zeros(b_shape),
            input_cache: None,
        }
    }
}

impl Module for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // input: [batch, in_features]
        self.input_cache = Some(input.clone());
        // TODO: implement actual matmul + bias
        let batch = input.shape[0];
        Tensor::zeros(&[batch, self.out_features])
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.input_cache.take().expect("No input cache");
        // TODO: compute gradients and grad_input
        Tensor::zeros(&input.shape)
    }

    fn update(&mut self, optimizer: &mut dyn Optimizer) {
        optimizer.step(
            &mut [&mut self.weight, &mut self.bias],
            &[&self.grad_weight, &self.grad_bias],
        );
    }
}