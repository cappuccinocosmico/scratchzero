use crate::mnist::tensor::Tensor;
use std::path::Path;

/// MNIST dataset
pub struct MnistDataset {
    /// Images tensor: [N, 1, 28, 28]
    pub images: Tensor,
    /// Labels: length N, values 0-9
    pub labels: Vec<u8>,
}

impl MnistDataset {
    /// Load MNIST dataset from given directory. Expects `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` files.
    pub fn load<P: AsRef<Path>>(dir: P) -> Self {
        // TODO: parse IDX files
        unimplemented!("MNIST loading not implemented");
    }
}
