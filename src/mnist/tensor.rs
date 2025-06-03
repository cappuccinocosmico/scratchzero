use smallvec::{SmallVec, ToSmallVec};

const MAX_TENSOR_DIMENSION: usize = 5;

/// N-dimensional tensor of `f32`.
pub struct TensorRef {
    /// Shape, e.g., [batch, channels, height, width].
    pub shape: SmallVec<[usize; MAX_TENSOR_DIMENSION]>,
    /// Flattened data in row-major order.
    pub data: [f32],
}

pub struct OwnedTensor {
    /// Shape, e.g., [batch, channels, height, width].
    pub shape: SmallVec<[usize; MAX_TENSOR_DIMENSION]>,
    /// Flattened data in row-major order.
    data: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct WrongDimensionError;

impl OwnedTensor {
    /// Creates a new tensor with given shape, filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        OwnedTensor {
            data: vec![0.0; size],
            shape: shape.to_smallvec(),
        }
    }

    /// Creates a new tensor from data and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        OwnedTensor {
            data,
            shape: shape.to_smallvec(),
        }
    }
}

impl TensorRef {
    // Could you create a function that takes in a reference to some data, and a &shape, that outputs a reference to a &Tensor, which has the data and shape.
    pub fn new_from_references<'a>(data: &'a [f32], shape: &[usize]) -> &Self {}

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &TensorRef) -> Result<OwnedTensor, WrongDimensionError> {
        if self.shape != other.shape {
            return Err(WrongDimensionError);
        };
        let mut result = OwnedTensor::zeros(&self.shape);
        for index in 0..self.len() {
            result.data[index] = self.data[index] + other.data[index]
        }
        Ok(result)
    }

    pub fn matmul(&self, other: &TensorRef) -> Result<OwnedTensor, WrongDimensionError> {
        // Check dimensionality (max 2D)
        if self.shape.len() > 2 || other.shape.len() > 2 {
            return Err(WrongDimensionError);
        }

        match (self.shape.len(), other.shape.len()) {
            // Matrix multiplication (MxN * NxP -> MxP)
            (2, 2) => {
                if self.shape[1] != other.shape[0] {
                    return Err(WrongDimensionError);
                }
                let m = self.shape[0];
                let n = self.shape[1];
                let p = other.shape[1];
                let mut result_data = vec![0.0; m * p];

                for i in 0..m {
                    for j in 0..p {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += self.data[i * n + k] * other.data[k * p + j];
                        }
                        result_data[i * p + j] = sum;
                    }
                }
                Ok(OwnedTensor::from_vec(result_data, &[m, p]))
            }

            // Matrix * Vector (MxN * N -> M)
            (2, 1) => {
                if self.shape[1] != other.shape[0] {
                    return Err(WrongDimensionError);
                }
                let m = self.shape[0];
                let n = self.shape[1];
                let mut result_data = vec![0.0; m];

                for i in 0..m {
                    for k in 0..n {
                        result_data[i] += self.data[i * n + k] * other.data[k];
                    }
                }
                Ok(OwnedTensor::from_vec(result_data, &[m]))
            }

            // Vector * Matrix (1xN * NxP -> 1xP treated as P)
            (1, 2) => {
                if self.shape[0] != other.shape[0] {
                    return Err(WrongDimensionError);
                }
                let n = self.shape[0];
                let p = other.shape[1];
                let mut result_data = vec![0.0; p];

                for j in 0..p {
                    for k in 0..n {
                        result_data[j] += self.data[k] * other.data[k * p + j];
                    }
                }
                Ok(OwnedTensor::from_vec(result_data, &[p]))
            }

            // Invalid combination
            _ => Err(WrongDimensionError),
        }
    }
}
