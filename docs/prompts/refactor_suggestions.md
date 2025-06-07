# Refactor Suggestions for MNIST CNN in Rust

Here’s one way you might organize your MNIST-CNN in Rust, carving it into clean modules, fully generic tensor types, and a lean functional-style `Module` trait that works equally well for inference and batched training.

---

# 1. Core Data Types

## 1.1 Tensor

A truly generic _N_-dimensional tensor that lives on the CPU and carries no side-state:
```rust
// src/tensor.rs
pub type Dim = usize;
pub type Float = f32;

/// An N-dimensional, row-major f32 array.
#[derive(Clone, Debug)]
pub struct Tensor<const D: usize> {
    /// e.g. [batch, channels, height, width] for D=4.
    pub shape: [Dim; D],
    pub data: Vec<Float>,
}

impl<const D: usize> Tensor<D> {
    /// Allocate uninitialized (or zeroed).
    pub fn zeros(shape: [Dim; D]) -> Self {
        let n = shape.iter().product();
        Tensor { shape, data: vec![0.0; n] }
    }

    /// Basic indexing, reshape, matmul (when D=2), etc.
    /// …
}
```

### 1.2 Parameter + Gradient Bundles

Rather than stuffing gradients in the `Tensor` itself, we wrap parameters in a small struct so that training-only code can manage them:
```rust
// src/field.rs
use crate::tensor::Tensor;

/// A “leaf” in the computational graph: holds the _value_ plus an accumulated _gradient_.
#[derive(Clone)]
pub struct Field<const D: usize> {
    pub value: Tensor<D>,
    pub grad:  Tensor<D>,
}

impl<const D: usize> Field<D> {
    pub fn new(value: Tensor<D>) -> Self {
        let grad = Tensor::zeros(value.shape);
        Field { value, grad }
    }

    pub fn zero_grad(&mut self) {
        self.grad.data.fill(0.0);
    }
}
```

---

# 2. The `Module` Trait

We keep every layer purely functional: forward returns `(output, cache)`, backward returns `(d_input, parameter_grad)`. Stateless layers simply use `()` as their `Cache` or `ParamGrad`.

```rust
// src/module.rs
use crate::tensor::Tensor;
use crate::field::Field;

/// Each layer or model that has (optional) parameters implements `Module`.
pub trait Module {
    /// The “shape” of inputs this layer consumes.
    type Input;
    /// The “shape” of outputs this layer produces.
    type Output;
    /// The per-layer parameters bundled with gradients.
    type Param;
    /// The shape of the parameter gradients (usually same as `Param`).
    type ParamGrad;
    /// Any cached activations needed for backward.
    type Cache;

    /// Pure forward: returns output _and_ whatever cache you need.
    fn forward(&self, input: &Self::Input)
        -> (Self::Output, Self::Cache);

    /// Pure backward: given grad w.r.t. output + the cache, returns
    /// - d_input: gradient w.r.t. input, and
    /// - param_grad: gradient w.r.t. this layer’s parameters.
    fn backward(
        &self,
        grad_output: &Self::Output,
        cache: &Self::Cache
    ) -> (Self::Input, Self::ParamGrad);

    /// Apply an optimizer update to the parameters.
    fn update(&mut self, param_grad: &Self::ParamGrad, lr: f32);
}
```

---

# 3. Layer Implementations

Each layer lives in `src/layers/`. They all follow the same pattern:
- **Param**: a tuple of `Field<…>`s
- **ParamGrad**: the raw `Tensor<…>` shapes
- **Cache**: whatever you need (indices for max-pool, im2col buffers, or `()` for ReLU)
- `forward` clones purely functional data, never mutates internal state.
- `backward` consumes `cache` + `grad_output` → `(grad_input, param_grad)`
- `update` applies the gradient via SGD (or momentum, Adam, etc.)

### 3.1 Linear (Dense)
```rust
// src/layers/linear.rs
use crate::{
    tensor::Tensor,
    field::Field,
    module::Module,
};

pub struct Linear {
    pub w: Field<2>,  // shape: [out_features, in_features]
    pub b: Field<1>,  // shape: [out_features]
}

pub struct LinearCache<const D: usize> {
    input: Tensor<D>,
}

impl Module for Linear {
    type Input     = Tensor<2>;  // [batch, in_features]
    type Output    = Tensor<2>;  // [batch, out_features]
    type Param     = (Tensor<2>, Tensor<1>);
    type ParamGrad = (Tensor<2>, Tensor<1>);
    type Cache     = LinearCache<2>;

    fn forward(&self, input: &Self::Input) -> (Self::Output, Self::Cache) {
        let batch = input.shape[0];
        // out = input.matmul(w^T) + b
        let mut out = Tensor::zeros([batch, self.w.value.shape[0]]);
        // … implement matmul + broadcast bias …
        (out.clone(), LinearCache { input: input.clone() })
    }

    fn backward(
        &self,
        grad_out: &Self::Output,
        cache: &Self::Cache,
    ) -> (Self::Input, Self::ParamGrad) {
        let x = &cache.input;
        // dW = grad_out^T.matmul(x)
        // dx = grad_out.matmul(w)
        // dB = grad_out.sum(axis=0)
        let dW = /* … */;
        let dB = /* … */;
        let dx = /* … */;
        (dx, (dW, dB))
    }

    fn update(&mut self, (dW, dB): &Self::ParamGrad, lr: f32) {
        // SGD step
        for (w, dw) in self.w.value.data.iter_mut().zip(&dW.data) {
            *w -= lr * dw;
        }
        for (b, db) in self.b.value.data.iter_mut().zip(&dB.data) {
            *b -= lr * db;
        }
    }
}
```

### 3.2 Conv2D, Pooling, ReLU, Softmax+Loss
Each of these lives under `src/layers/`. They share exactly the same pattern described above.

---

# 4. Model & Composition

A simple `Sequential` container:
```rust
// src/models/sequential.rs
use crate::module::Module;

pub struct Sequential<Layers>(pub Vec<Box<dyn Module<Input=…, Output=…, …>>>);

impl<…> Module for Sequential<…> {
    /* forward loops through layers, collecting caches in a Vec */
    /* backward unrolls in reverse, threading grads + layer-by-layer param_grads */
    /* update calls update() on each layer */
}
```

You may also write a rich `Model` struct that owns a `Sequential<[Conv2D, ReLU, Pool, …]>` and wraps a CrossEntropy loss in its `forward_and_loss()` method.

---

# 5. Optimizers

Under `src/optim/` you can define traits:
```rust
pub trait Optimizer {
    type ParamGrad;
    fn step(&mut self, param: &mut impl ParamAccess, grad: &Self::ParamGrad);
}
```

And concrete impls like SGD, Adam, etc.


# 8. Why This Design?

- **Zero runtime state** in your layers: caches are entirely ephemeral, so you can batch, parallelize, or share a model immutably.
- **Pure functions** for forward/backward → easy to test, parallelize, and reason about.
- **Explicit grads**: your optimizer sees exactly the `ParamGrad` it needs, and you never accidentally mix “last batch” state with current.
- **Minimal Trait**: only one `Module` trait to learn—you get swap-in/out of ReLU, Conv2D, Linear, even Dropout, without changing the loop.

From here you can:
- Add `const` generics for well-known shapes (e.g. `Tensor<4>` for images).
- Build a v2 with GPU backends by swapping the `Tensor` impl behind a `TensorTrait`.
- Extend the optimizer API to something like `Adam` without touching your model code.

This should give you a clean, idiomatic Rust foundation for both **inference** and **training** on MNIST (or anything else you plug in).
