use crate::mnist::{
    field::Field,
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
        let in_features = input.shape()[1];
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
        let in_features = input.shape()[1];
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
        let batch = grad_output.shape()[0];
        let in_features = cache.input.shape[1];
        let out_features = grad_output.shape[1];
        // dW shape [out_features, in_features]
        let mut dW = Tensor::<2>::zeros([out_features, in_features]);
        // dB shape [out_features]
        let mut dB = Tensor::<1>::zeros([out_features]);
        // dX shape [batch, in_features]
        let mut dX = Tensor::<2>::zeros([batch, in_features]);

        for b_idx in 0..batch {
            for j in 0..out_features {
                let go = grad_output.data()[b_idx * out_features + j];
                dB.data_mut()[j] += go;
                for k in 0..in_features {
                    dW.data_mut()[j * in_features + k] +=
                        go * cache.input.data()[b_idx * in_features + k];
                    dX.data_mut()[b_idx * in_features + k] +=
                        go * self.w.value.data()[j * in_features + k];
                }
            }
        }
        (dX, (dW, dB))
    }

    fn update(&mut self, param_grad: &Self::Param, lr: f32) {
        let (ref dW, ref dB) = *param_grad;
        // SGD update
        for (w_val, dw) in self.w.value.data_mut().iter_mut().zip(dW.data().iter()) {
            *w_val -= lr * dw;
        }
        for (b_val, db) in self.b.value.data_mut().iter_mut().zip(dB.data().iter()) {
            *b_val -= lr * db;
        }
    }
}
