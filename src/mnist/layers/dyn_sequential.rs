use std::any::{Any, TypeId};

use crate::mnist::{
    layers::{
        dyn_module::{DynModule, dynify},
        relu::ReLU,
    },
    module::ModConfig,
    types::{Module, Tensor},
};

pub trait SequenceConfig {
    type Input;
    type Output;
}

pub struct Sequential<T: SequenceConfig> {
    config: T,
    layers: Vec<Box<dyn DynModule>>,
}

impl<T: SequenceConfig> ModConfig for Sequential<T> {
    type Input = T::Input;
    type Output = T::Output;
    type Param = Vec<Box<dyn Any>>;
    type Cache = Vec<Box<dyn Any>>;
}

impl<T: SequenceConfig + 'static> Sequential<T> {
    fn new(layers: Vec<Box<dyn DynModule>>, config: T) -> Option<Self> {
        if layers.is_empty() {
            return None;
        };
        if TypeId::of::<T::Input>() != layers[0].input_type() {
            return None;
        };
        if TypeId::of::<T::Output>() != layers[layers.len() - 1].output_type() {
            return None;
        };
        for index in 0..layers.len() - 1 {
            if layers[index].output_type() != layers[index + 1].input_type() {
                return None;
            }
        }
        Some(Sequential { config, layers })
    }
}

impl<T: SequenceConfig + 'static> Module for Sequential<T> {
    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache) {
        let mut current_input: &dyn Any = input;
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut last_output: Box<dyn Any> = Box::new(());

        for layer in self.layers.iter() {
            let forward_boxes = layer.forward_boxed(current_input);
            caches.push(forward_boxes.cache);
            last_output = forward_boxes.output;
            current_input = last_output.as_ref();
        }

        let output = *last_output.downcast::<T::Output>().unwrap();
        (output, caches)
    }
    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::Param) {
        let layer_len = self.layers.len();
        let mut current_grad_output: &dyn Any = grad_output;
        let mut last_grad_input: Box<dyn Any> = Box::new(());
        let mut grad_params: Vec<Box<dyn Any>> = (0..layer_len)
            .map(|_| Box::new(()) as Box<dyn Any>)
            .collect();

        for reverse_index in 0..layer_len {
            let index = layer_len - reverse_index - 1;
            let layer = &self.layers[index];
            let backward_boxes = layer.backward_boxed(current_grad_output, cache[index].as_ref());
            grad_params[index] = backward_boxes.grad_params;
            last_grad_input = backward_boxes.grad_input;
            current_grad_output = last_grad_input.as_ref();
        }

        let input = *last_grad_input.downcast::<T::Input>().unwrap();
        (input, grad_params)
    }
    fn update(&mut self, param_grad: &Self::Param, lr: f32) {
        for (index, layer_box) in self.layers.iter_mut().enumerate() {
            let layer = layer_box.as_mut();
            layer.update_boxed(param_grad[index].as_ref(), lr);
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct TestConfig {}
impl SequenceConfig for TestConfig {
    type Input = Tensor<1>;
    type Output = Tensor<1>;
}

pub fn make_test_sequential() -> Sequential<TestConfig> {
    Sequential::new(vec![dynify(ReLU::<1>)], TestConfig::default()).unwrap()
}
