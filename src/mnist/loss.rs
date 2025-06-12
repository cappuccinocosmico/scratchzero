use crate::mnist::tensor::Tensor;

/// Computes softmax cross-entropy loss and gradient w.r.t. logits.
///
/// pred: Tensor<1> of shape [num_classes], raw logits.
/// label: ground truth index in [0, num_classes).
/// Returns (loss, grad_logits)
pub fn softmax_cross_entropy(
    pred: &Tensor<1>,
    label: usize,
) -> (f32, Tensor<1>) {
    // numerical stability: shift by max
    let max_logit = pred.data().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = pred.data().iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let mut probs: Vec<f32> = exp_vals.iter().map(|&v| v / sum_exp).collect();
    let loss = -probs[label].ln();
    // gradient of loss w.r.t. logits: probs - one_hot
    probs[label] -= 1.0;
    let grad = Tensor::from_vec_unchecked(probs, *pred.shape());
    (loss, grad)
}

/// Mean Squared Error loss and gradient.
///
/// pred: Tensor<1>
/// target: Tensor<1>
/// Returns (loss, grad_pred)
pub fn mse(
    pred: &Tensor<1>,
    target: &Tensor<1>,
) -> (f32, Tensor<1>) {
    let n = pred.len() as f32;
    let mut grad = Tensor::<1>::zeros(*pred.shape());
    let mut loss = 0.0;
    for i in 0..pred.len() {
        let diff = pred.data()[i] - target.data()[i];
        loss += diff * diff;
        grad.data_mut()[i] = 2.0 * diff / n;
    }
    loss /= n;
    (loss, grad)
}
