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


However there seem to be a couple problems with these types. Namely for tensors after actually writing the code it makes sense to embed the dimensionality of a tensor in the code as a generic constant like so:
```rs

type DimSizeType = usize;
type flt = f32;

pub struct Tensor<const dim: usize> {
    /// Shape, e.g., [batch, channels, height, width].
    pub shape: [DimSizeType; dim],
    /// Flattened data in row-major order.
    data: Vec<flt>,
}
```

I also went ahead and implemented by hand all of the matrix multiplication and other tools. You can find it all in:
/home/nicole/Documents/mycorrhizae/scratchzero/src/mnist/tensor.rs

As for modules, I am still thinking a bit about what to do. Mainly, even though it might be possible to store the cache results internally, that would totally interfere with the ability to do batch processing and apply the weights all at once, and also make it impossible to use the model for inference across multiple threads, and also kind of goes against rust's functional programming ethos. However I dont necessarially know how to implement this. Here was one of my attempts:

```rust
/// Core trait for layers and models.
pub trait Module {
    type InternalData;
    type CacheData;
    type InputData;
    type OutputData;
    /// Forward pass with cache: input -> output, cache.
    fn forward_with_cache(&self, input: &Self::InputData) -> (Self::OutputData, Self::CacheData);
    /// Forward pass without cache.
    fn forward(&self, input: &Self::InputData) -> Self::OutputData {
        self.forward_with_cache(input).0
    }
    /// Backward pass: gradient of output -> gradient of input, gradient of layer weights
    fn backward(
        &self,
        grad_output: &Self::OutputData,
        cache: &Self::CacheData,
    ) -> (Self::InputData, Self::InternalData);
    /// Apply an offset to the weights.
    fn apply_offset_to_weights(&mut self, offset: &Self::InternalData);
}
```

However this would still present some problems. Namely there is some inefficences because it still needs to cache results for ReLU layers that dont need any backpropogation. 

There also needs to be some kind of trait for the internal data that would let you aggregate multiple elements together to apply results from a batch process. But all it should need is an ability to add together the types, and do scalar multiplication for stochastic gradient descent.


And I also have a suspicion that there might be a way to implement this more simply while retaining all the functionality thoughts?
