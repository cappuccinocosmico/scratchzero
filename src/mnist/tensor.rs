use smallvec::{SmallVec, ToSmallVec};

const MAX_TENSOR_DIMENSION: usize = 5;
type DimSizeType = usize;
type flt = f32;

pub struct Tensor<const dim: usize> {
    /// Shape, e.g., [batch, channels, height, width].
    pub shape: [DimSizeType; dim],
    /// Flattened data in row-major order.
    data: Vec<flt>,
}

#[derive(Clone, Copy, Debug)]
pub struct WrongDimensionError;
impl<const dim: usize> Tensor<dim> {
    /// Mutable access to underlying data.
    pub fn data_mut(&mut self) -> &mut [flt] {
        &mut self.data
    }

    /// Immutable access to underlying data.
    pub fn data(&self) -> &[flt] {
        &self.data
    }

    
    pub fn tnsr_prod<const odim: DimSizeType>(
        &self,
        other: &Tensor<odim>,
    ) -> Tensor<{ dim + odim }> {
        // Calculate new shape by concatenating dimensions
        let mut new_shape = [0; dim + odim];
        new_shape[..dim].copy_from_slice(&self.shape);
        new_shape[dim..].copy_from_slice(&other.shape);

        // Compute outer product values
        let mut data = Vec::with_capacity(self.data.len() * other.data.len());
        for &a in &self.data {
            for &b in &other.data {
                data.push(a * b);
            }
        }

        Tensor::from_vec_unchecked(data, new_shape)
    }
    /// Creates a new tensor with given shape, filled with zeros.
    pub fn zeros(shape: [usize; dim]) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Creates a new tensor from data and shape.
    pub fn from_vec(
        data: Vec<flt>,
        shape: [DimSizeType; dim],
    ) -> Result<Self, WrongDimensionError> {
        if data.len() != shape.iter().product() {
            return Err(WrongDimensionError);
        }
        Ok(Self::from_vec_unchecked(data, shape))
    }
    pub fn from_vec_unchecked(data: Vec<flt>, shape: [usize; dim]) -> Self {
        Tensor { data, shape }
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Tensor<dim>) -> Result<Tensor<dim>, WrongDimensionError> {
        if self.shape != other.shape {
            return Err(WrongDimensionError);
        };

        let mut data = Vec::with_capacity(self.data.len());
        for index in 0..self.len() {
            data.push(self.data[index] + other.data[index])
        }
        Ok(Self::from_vec_unchecked(data, self.shape))
    }
}
impl Tensor<2> {
    pub fn matmul(&self, other: &Tensor<2>) -> Result<Tensor<2>, WrongDimensionError> {
        if self.shape[1] != other.shape[0] {
            return Err(WrongDimensionError);
        }
        let m = self.shape[0];
        let n = self.shape[1];
        let p = other.shape[1];
        let mut result_data = vec![0.0; m * p];
        // TODO: Implement the faster strassen matmul algorithm
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.data[i * n + k] * other.data[k * p + j];
                }
                result_data[i * p + j] = sum;
            }
        }
        Ok(Tensor::<2>::from_vec(result_data, [m, p]).unwrap())
    }
    pub fn vecmul(&self, other: &Tensor<1>) -> Result<Tensor<1>, WrongDimensionError> {
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
        Ok(Tensor::<1>::from_vec_unchecked(result_data, [m]))
    }
}

impl Tensor<1> {
    pub fn dot(&self, other: &Tensor<1>) -> Result<flt, WrongDimensionError> {
        if self.shape[0] != other.shape[0] {
            return Err(WrongDimensionError);
        };
        let mut collector = 0.0;
        for i in 0..self.shape[0] {
            collector += self.data[i] * other.data[i]
        }
        Ok(collector)
    }
}
