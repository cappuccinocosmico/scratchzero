/// N-dimensional tensor of `f32`.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Flattened data in row-major order.
    pub data: Vec<f32>,
    /// Shape, e.g., [batch, channels, height, width].
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new tensor with given shape, filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    /// Creates a new tensor from data and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product());
        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Reshapes the tensor (must match total size).
    pub fn reshape(&mut self, new_shape: &[usize]) {
        assert_eq!(new_shape.iter().product::<usize>(), self.len());
        self.shape = new_shape.to_vec();
    }

    /// Index into tensor (row-major) for reading.
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Mutable index into tensor.
    pub fn get_mut(&mut self, idx: usize) -> &mut f32 {
        &mut self.data[idx]
    }

    // TODO: implement more operations: matmul, add, etc.
}
