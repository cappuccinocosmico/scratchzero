use crate::mnist::{
    module::{ModConfig, Module},
    tensor::Tensor,
};

/// ReLU activation layer.
#[derive(Clone, Copy, Default)]
pub struct ReLU<const D: usize>;

impl<const D: usize> ReLU<D> {
    pub fn new() -> Self {
        ReLU
    }
}

pub struct ReLUCache<const D: usize> {
    pub input: Tensor<D>,
}

impl<const D: usize> ModConfig for ReLU<D> {
    type Input = Tensor<D>;
    type Output = Tensor<D>;
    type Param = (); // no parameters
    type Cache = ReLUCache<D>;
}
impl<const D: usize> Module for ReLU<D> {
    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache) {
        let mut output = input.clone();
        for v in output.data_mut().iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        (
            output.clone(),
            ReLUCache {
                input: input.clone(),
            },
        )
    }

    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::Param) {
        let mut grad_input = grad_output.clone();
        for (i, v) in cache.input.data().iter().enumerate() {
            if *v <= 0.0 {
                grad_input.data_mut()[i] = 0.0;
            }
        }
        (grad_input, ())
    }

    fn update(&mut self, _param_grad: &Self::Param, _lr: f32) {
        // no parameters to update
    }
}
