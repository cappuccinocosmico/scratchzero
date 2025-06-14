use crate::mnist::{
    module::{ModConfig, Module},
    tensor::Tensor,
};

/// 2D Convolutional Layer (no dilation, no groups)
pub struct Conv2d {
    pub weight: Tensor<4>, // [out_channels, in_channels, kernel_h, kernel_w]
    pub bias: Tensor<1>,   // [out_channels]
    pub stride: usize,
    pub padding: usize,
}

/// Cache for Conv2d layer
pub struct Conv2dCache {
    pub input: Tensor<3>,        // [in_channels, in_h, in_w]
    pub padded_input: Tensor<3>, // input after padding
}

impl Conv2d {
    /// Create a new Conv2d layer with random weights in [-1,1]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let weight = Tensor::<4>::random([out_channels, in_channels, kernel_h, kernel_w]);
        let bias = Tensor::<1>::random([out_channels]);
        Conv2d {
            weight,
            bias,
            stride,
            padding,
        }
    }

    fn pad_input(input: &Tensor<3>, pad: usize) -> Tensor<3> {
        let [in_c, in_h, in_w] = *input.shape();
        let padded_h = in_h + 2 * pad;
        let padded_w = in_w + 2 * pad;
        let mut padded = Tensor::<3>::zeros([in_c, padded_h, padded_w]);
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let src_idx = c * (in_h * in_w) + h * in_w + w;
                    let dst_idx = c * (padded_h * padded_w) + (h + pad) * padded_w + (w + pad);
                    padded.data_mut()[dst_idx] = input.data()[src_idx];
                }
            }
        }
        padded
    }
}

impl ModConfig for Conv2d {
    type Input = Tensor<3>;
    type Output = Tensor<3>;
    type Param = (Tensor<4>, Tensor<1>);
    type Cache = Conv2dCache;
}

impl Module for Conv2d {
    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache) {
        let [in_c, in_h, in_w] = *input.shape();
        let [out_c, _, k_h, k_w] = *self.weight.shape();
        let stride = self.stride;
        let pad = self.padding;
        let padded = Conv2d::pad_input(input, pad);
        let [_, p_h, p_w] = *padded.shape();
        let out_h = (p_h - k_h) / stride + 1;
        let out_w = (p_w - k_w) / stride + 1;
        let mut output = Tensor::<3>::zeros([out_c, out_h, out_w]);
        // Convolution
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    for ic in 0..in_c {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let inp_idx = ic * (p_h * p_w) + ih * p_w + iw;
                                let w_idx =
                                    oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                                sum += padded.data()[inp_idx] * self.weight.data()[w_idx];
                            }
                        }
                    }
                    sum += self.bias.data()[oc];
                    let out_idx = oc * (out_h * out_w) + oh * out_w + ow;
                    output.data_mut()[out_idx] = sum;
                }
            }
        }
        (
            output,
            Conv2dCache {
                input: input.clone(),
                padded_input: padded,
            },
        )
    }

    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::Param) {
        let input = &cache.input;
        let padded = &cache.padded_input;
        let [in_c, in_h, in_w] = *input.shape();
        let [out_c, _, k_h, k_w] = *self.weight.shape();
        let stride = self.stride;
        let pad = self.padding;
        let [_, p_h, p_w] = *padded.shape();
        let [_, out_h, out_w] = *grad_output.shape();
        // grads
        let mut grad_input_padded = Tensor::<3>::zeros([in_c, p_h, p_w]);
        let mut grad_weight = Tensor::<4>::zeros(*self.weight.shape());
        let mut grad_bias = Tensor::<1>::zeros(*self.bias.shape());
        // compute
        for oc in 0..out_c {
            // bias grad
            let mut db = 0.0;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let go = grad_output.data()[oc * (out_h * out_w) + oh * out_w + ow];
                    db += go;
                }
            }
            grad_bias.data_mut()[oc] = db;
            for ic in 0..in_c {
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let mut dw = 0.0;
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let inp = padded.data()[ic * (p_h * p_w) + ih * p_w + iw];
                                let go = grad_output.data()[oc * (out_h * out_w) + oh * out_w + ow];
                                dw += inp * go;
                                // grad input convolution
                                let w_val = self.weight.data()
                                    [oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw];
                                grad_input_padded.data_mut()[ic * (p_h * p_w) + ih * p_w + iw] +=
                                    w_val * go;
                            }
                        }
                        grad_weight.data_mut()
                            [oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw] = dw;
                    }
                }
            }
        }
        // remove padding from grad_input_padded
        let mut grad_input = Tensor::<3>::zeros([in_c, in_h, in_w]);
        for ic in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let src =
                        grad_input_padded.data()[ic * (p_h * p_w) + (h + pad) * p_w + (w + pad)];
                    grad_input.data_mut()[ic * (in_h * in_w) + h * in_w + w] = src;
                }
            }
        }
        (grad_input, (grad_weight, grad_bias))
    }

    fn update(&mut self, param_grad: &Self::Param, lr: f32) {
        let (ref gw, ref gb) = *param_grad;
        // update weight
        for (w_val, dw) in self.weight.data_mut().iter_mut().zip(gw.data().iter()) {
            *w_val -= lr * *dw;
        }
        // update bias
        for (b_val, db) in self.bias.data_mut().iter_mut().zip(gb.data().iter()) {
            *b_val -= lr * *db;
        }
    }
}

