use crate::mnist::tensor::Tensor;

/// A parameter with value and accumulated gradient.
#[derive(Clone, Debug)]
pub struct Field<const D: usize> {
    pub value: Tensor<D>,
    pub grad: Tensor<D>,
}

impl<const D: usize> Field<D> {
    /// Create a new Field with zero-initialized gradients.
    pub fn new(value: Tensor<D>) -> Self {
        let grad = Tensor::zeros(value.shape);
        Field { value, grad }
    }

    /// Zero out the accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.grad.data_mut().iter_mut().for_each(|v| *v = 0.0);
    }
}

