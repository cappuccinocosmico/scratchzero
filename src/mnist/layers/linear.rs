use crate::mnist::{
    module::{ModConfig, Module},
    tensor::Tensor,
};

/// Linear (Fully Connected) Layer
pub struct Linear {
    pub w: Tensor<2>, // [out_features, in_features]
    pub b: Tensor<1>, // [out_features]
}

/// Cache for Linear layer (stores input)
pub struct LinearCache {
    pub input: Tensor<1>,
}

impl Linear {
    /// Create a new Linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let w = Tensor::<2>::random([out_features, in_features]);
        let b = Tensor::<1>::random([out_features]);
        Linear { w, b }
    }
}

impl ModConfig for Linear {
    type Input = Tensor<1>;
    type Output = Tensor<1>;
    type Param = (Tensor<2>, Tensor<1>);
    type Cache = LinearCache;
}

impl Module for Linear {
    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache) {
        let in_features = input.shape()[0];
        let out_features = self.b.shape()[0];
        let mut out = Tensor::<1>::zeros([out_features]);
        // out[b, j] = sum_k input[b, k] * w[j, k] + b[j]
        for j in 0..out_features {
            let mut sum = 0.0;
            for k in 0..in_features {
                sum += input.data()[k] * self.w.data()[j * in_features + k];
            }
            sum += self.b.data()[j];
            out.data_mut()[j] = sum;
        }
        (
            out,
            LinearCache {
                input: input.clone(),
            },
        )
    }

    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::Param) {
        let input = &cache.input;
        let in_features = input.shape()[0];
        let out_features = grad_output.shape()[0];
        // grad_input: [in_features]
        let mut grad_input = Tensor::<1>::zeros([in_features]);
        // grad_w: [out_features, in_features]
        let mut grad_w = Tensor::<2>::zeros([out_features, in_features]);
        // grad_b: [out_features]
        let mut grad_b = Tensor::<1>::zeros([out_features]);
        // Compute gradients
        for j in 0..out_features {
            let grad_out_j = grad_output.data()[j];
            // bias gradient
            grad_b.data_mut()[j] = grad_out_j;
            for k in 0..in_features {
                // weight gradient
                let idx = j * in_features + k;
                grad_w.data_mut()[idx] = grad_out_j * input.data()[k];
                // input gradient
                grad_input.data_mut()[k] += self.w.data()[idx] * grad_out_j;
            }
        }
        (grad_input, (grad_w, grad_b))
    }

    fn update(&mut self, param_grad: &Self::Param, lr: f32) {
        let (ref grad_w, ref grad_b) = *param_grad;
        // Update weights
        for (w_val, gw) in self.w.data_mut().iter_mut().zip(grad_w.data().iter()) {
            *w_val -= lr * *gw;
        }
        // Update biases
        for (b_val, gb) in self.b.data_mut().iter_mut().zip(grad_b.data().iter()) {
            *b_val -= lr * *gb;
        }
    }
}
