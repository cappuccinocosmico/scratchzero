// use crate::mnist::tensor::Tensor;
//
// /// Softmax Cross-Entropy loss.
// /// Returns (loss, gradient wrt logits)
// pub fn softmax_cross_entropy(
//     logits: &Tensor,
//     labels: &Tensor, // one-hot [batch, num_classes]
// ) -> (f32, Tensor) {
//     let batch = logits.shape[0];
//     let classes = logits.shape[1];
//     let mut total_loss = 0.0;
//     let mut grad_data = vec![0.0; logits.data.len()];
//
//     for i in 0..batch {
//         let logits_slice = &logits.data[i*classes..(i+1)*classes];
//         let labels_slice = &labels.data[i*classes..(i+1)*classes];
//
//         // Numerical stabilization: subtract max logit
//         let max_logit = logits_slice.iter()
//             .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
//
//         // Compute exponents and softmax
//         let mut exponents: Vec<f32> = logits_slice.iter()
//             .map(|x| (x - max_logit).exp())
//             .collect();
//         let sum_exponents: f32 = exponents.iter().sum();
//         let softmax: Vec<f32> = exponents.iter()
//             .map(|x| x / sum_exponents)
//             .collect();
//
//         // Compute loss contribution for this example
//         let true_class = labels_slice.iter()
//             .position(|&x| x == 1.0)
//             .expect("Labels must be one-hot encoded");
//         total_loss += -softmax[true_class].ln();
//
//         // Compute gradient for this example
//         for j in 0..classes {
//             let grad_idx = i * classes + j;
//             grad_data[grad_idx] = (softmax[j] - labels_slice[j]) / batch as f32;
//         }
//     }
//
//     // Average loss over batch
//     total_loss /= batch as f32;
//
//     (total_loss, Tensor::(grad_data, logits.shape.clone()))
// }
