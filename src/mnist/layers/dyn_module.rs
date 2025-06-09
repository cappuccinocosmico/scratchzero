use std::any::{Any, TypeId};

use crate::mnist::{layers::relu::ReLU, module::ModConfig, types::Module};

/// A module that takes boxed `Any` as input/output.
/// We require all params/caches also be `Any + 'static`.
pub struct BackwardBoxes {
    pub grad_input: Box<dyn Any>,
    pub grad_params: Box<dyn Any>,
}

pub struct ForwardBoxes {
    pub output: Box<dyn Any>,
    pub cache: Box<dyn Any>,
}
pub trait DynModule {
    /// The concrete input and output types
    fn input_type(&self) -> TypeId;
    fn output_type(&self) -> TypeId;

    /// Forward: takes a boxed input, returns (boxed output, boxed cache)
    fn forward_boxed(&self, input: &dyn Any) -> ForwardBoxes;

    /// Backward: takes boxed grad_output and cache,
    /// returns (boxed grad_input, boxed grad_param)
    fn backward_boxed(&self, grad_output: &dyn Any, cache: &dyn Any) -> BackwardBoxes;

    /// Update parameters from boxed grad, given lr
    fn update_boxed(&mut self, grad_param: &dyn Any, lr: f32);
}

// pub trait ModConfig {
//     /// The type of input this module consumes.
//     type Input;
//     /// The type of output this module produces.
//     type Output;
//     /// The type of this module (pure values), and also the type of the gradient.
//     type Param;
//     /// Any cached activations needed for backward.
//     type Cache;
// impl<M> DynModule for ModuleWrapper<M>
impl<M> DynModule for M
where
    M: Module + ModConfig,
    M::Input: 'static,
    M::Output: 'static,
    M::Cache: 'static,
    M::Param: 'static,
{
    // 1. the associated type `<M as mnist::module::ModConfig>::Input` may not live long enough
    //    the associated type `<M as mnist::module::ModConfig>::Input` must be valid for the static lifetime... [E0310]
    fn input_type(&self) -> TypeId {
        TypeId::of::<M::Input>()
    }
    fn output_type(&self) -> TypeId {
        TypeId::of::<M::Output>()
    }

    fn forward_boxed(&self, input: &dyn Any) -> ForwardBoxes {
        // Downcast, run real forward, then box
        let x = input
            .downcast_ref::<M::Input>()
            .expect("input type mismatch");
        let (output, cache) = self.forward(x);

        ForwardBoxes {
            output: Box::new(output),
            cache: Box::new(cache),
        }
    }

    fn backward_boxed(&self, grad_output: &dyn Any, cache: &dyn Any) -> BackwardBoxes {
        let g = grad_output
            .downcast_ref::<M::Output>()
            .expect("grad_output type mismatch");
        let c = cache
            .downcast_ref::<M::Cache>()
            .expect("cache type mismatch");
        let (grad_input, grad_param) = self.backward(g, c);
        BackwardBoxes {
            grad_input: Box::new(grad_input),
            grad_params: Box::new(grad_param),
        }
    }

    fn update_boxed(&mut self, grad_param: &dyn Any, lr: f32) {
        let gp = grad_param
            .downcast_ref::<M::Param>()
            .expect("param grad type mismatch");
        self.update(gp, lr);
    }
}

pub fn dynify<M: Module + 'static>(module: M) -> Box<dyn DynModule> {
    Box::new(module)
}
