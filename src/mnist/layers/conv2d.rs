use crate::mnist::tensor::Tensor;
use crate::mnist::module::Module;
use crate::mnist::optim::Optimizer;

/// 2D convolution layer (no dilation, same padding support, stride=1)
pub struct Conv2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub weight: Tensor,     // [out, in, kh, kw]
    pub bias: Tensor,       // [out]
    // Gradients
    pub grad_weight: Tensor,
    pub grad_bias: Tensor,
    // Cache input for backward
    input_cache: Option<Tensor>,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kh: usize, kw: usize) -> Self {
        let w_shape = &[out_channels, in_channels, kh, kw];
        let b_shape = &[out_channels];
        Conv2D {
            in_channels,
            out_channels,
            kernel_size: (kh, kw),
            weight: Tensor::zeros(w_shape),
            bias: Tensor::zeros(b_shape),
            grad_weight: Tensor::zeros(w_shape),
            grad_bias: Tensor::zeros(b_shape),
            input_cache: None,
        }
    }
}

impl Module for Conv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // TODO: actual convolution op
        self.input_cache = Some(input.clone());
        // placeholder: return zeros
        let batch = input.shape[0];
        let height = input.shape[2];
        let width = input.shape[3];
        Tensor::zeros(&[batch, self.out_channels, height, width])
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.input_cache.take().expect("No input cache");
        // TODO: compute grad_weight, grad_bias, grad_input
        // placeholder: zeros
        Tensor::zeros(&input.shape)
    }

    fn update(&mut self, optimizer: &mut dyn Optimizer) {
        optimizer.step(&mut [ &mut self.weight, &mut self.bias ], &[ &self.grad_weight, &self.grad_bias ]);
    }
}