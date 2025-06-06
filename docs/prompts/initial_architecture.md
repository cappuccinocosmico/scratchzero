# Initial Architecture for MNIST Classification CNN in Rust

## Overview
This document outlines a high-level design for implementing a Convolutional Neural Network (CNN) to classify MNIST digits in Rust. We cover:

- Core data types for tensors, layers, models, and training artifacts.
- Module breakdown and organization.
- Data structures needed for inference and training.

---

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

