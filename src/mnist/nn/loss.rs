use crate::mnist::tensor::Tensor;

/// Softmax Cross-Entropy loss.
/// Returns (loss, gradient wrt logits)
pub fn softmax_cross_entropy(
    logits: &Tensor,
    labels: &Tensor,  // one-hot [batch, num_classes]
) -> (f32, Tensor) {
    let batch = logits.shape[0];
    let classes = logits.shape[1];
    // TODO: implement stable softmax and cross-entropy
    // placeholder loss=0, grad zeros
    (0.0, Tensor::zeros(&logits.shape))
}