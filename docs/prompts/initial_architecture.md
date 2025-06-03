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

### 2.1 Convolution2D
- Weights: `Tensor` of shape `[out_channels, in_channels, k_h, k_w]`
- Bias: `Tensor` of shape `[out_channels]`

### 2.2 Dense (Fully Connected)
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

Example CNN architecture:
```rust
let model = Sequential::new(vec![
    Conv2D::new(1, 32, 3, 1, 1),
    ReLU::new(),
    Conv2D::new(32, 64, 3, 1, 1),
    ReLU::new(),
    MaxPool2D::new(2, 2),
    Flatten::new(),
    Dense::new(64 * 14 * 14, 128),
    ReLU::new(),
    Dense::new(128, 10),
]);
```

---

## 4. Optimizers

```rust
pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);
}

pub struct SGD { learning_rate: f32 }
impl Optimizer for SGD { ... }
```

---

## 5. Data Loading

### 5.1 MNIST Dataset
```rust
pub struct MnistDataset {
    images: Tensor, // shape [num_samples, 1, 28, 28]
    labels: Vec<u8>,
}

pub struct DataLoader<'a> {
    dataset: &'a MnistDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
}
```

- Methods: `iter() -> Iterator<Item=(Tensor, Tensor)>` producing `(images_batch, labels_batch_one_hot)`.

---

## 6. Training Loop

```rust
fn train(
    model: &mut Sequential,
    optimizer: &mut dyn Optimizer,
    dataloader: &mut DataLoader,
    epochs: usize,
) {
    for epoch in 0..epochs {
        for (x, y) in dataloader.iter() {
            // Forward
            let logits = model.forward(&x);
            let (loss, grad_loss) = softmax_cross_entropy(&logits, &y);
            // Backward
            let grad = model.backward(&grad_loss);
            // Update
            model.update(optimizer);
        }
    }
}
```

---

## 7. Module Organization

- `core/`
  - `tensor.rs`
  - `module.rs`
- `layers/`
  - `conv2d.rs`
  - `dense.rs`
  - `pooling.rs`
  - `activation.rs`
- `nn/`
  - `sequential.rs`
  - `loss.rs`
- `optim/`
  - `sgd.rs`
  - `adam.rs`
- `data/`
  - `mnist.rs`
  - `dataloader.rs`
- `examples/`
  - `train_mnist.rs`

---

### Summary
This design balances flexibility, type safety, and idiomatic Rust patterns. We define clear module boundaries, generic tensor operations, and extensible traits for layers and optimizers. Subsequent steps include implementing and benchmarking each component, followed by potential GPU offloading.
