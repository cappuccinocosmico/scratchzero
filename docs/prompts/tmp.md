I had written out the broad architecture and goals for the project, as well as all the math for implementing the first component in 
/home/nicole/Documents/mycorrhizae/scratchzero/docs/initial.md

What would be your takes on how you could implement this initial mnist classiciation CNN in rust, could you go ahead and tryh to describe what data types we would need, both for inference and training. And then try to organize those into broader modules.

For the initial run of these prompts you suggested the following architecture:

## 1. Core Data Types

### 1.1 Tensor
A generic multi-dimensional array storing `f32` data on CPU memory.

```rust
/// N-dimensional tensor of `f32`.
pub struct Tensor {
    /// Flattened data in row-major order.
    data: Vec<f32>,
    /// Shape, e.g., [batch, channels, height, width].
    shape: Vec<usize>,
}
```

- Methods: `new(shape: &[usize])`, `zeros(...)`, `ones(...)`, `reshape(...)`, `indexing`, `to_device()`, `from_slice()`, `matmul()`, etc.

### 1.2 Module Trait
Shared interface for all neural layers/

```rust
/// Core trait for layers and models.
pub trait Module {
    /// Forward pass: input -> output.
    fn forward(&mut self, input: &Tensor) -> Tensor;
    /// For training: backward pass, returns gradient w.r.t input.
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    /// Update parameters using optimizer state.
    fn update(&mut self, optimizer: &mut dyn Optimizer);
}
```

---

## 2. Layer Types


### 2.1 Dense (Fully Connected)
- Weights: `Tensor` of shape `[out_features, in_features]`
- Bias: `Tensor` of shape `[out_features]`

### 2.3 Activation Functions
- ReLU: stateless layer implementing `Module`.
- Softmax + CrossEntropy: combines softmax plus loss.

---

## 3. Model Composition

```rust
/// Sequential container for stacking layers.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Module for Sequential {
    fn forward(&mut self, input: &Tensor) -> Tensor { ... }
    fn backward(&mut self, grad: &Tensor) -> Tensor { ... }
    fn update(&mut self, optimizer: &mut dyn Optimizer) { ... }
}
```



Once done write all your thoughts to 

/home/nicole/Documents/mycorrhizae/scratchzero/docs/prompts/initial_architecture.md
