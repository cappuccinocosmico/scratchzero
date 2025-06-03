use crate::mnist::data::{DataLoader, MnistDataset};
use crate::mnist::layers::{
    activation::Flatten, activation::ReLU, conv2d::Conv2D, dense::Dense, pooling::MaxPool2D,
};
use crate::mnist::nn::{Sequential, loss::softmax_cross_entropy};
use crate::mnist::optim::sgd::SGD;

fn main() {
    // Load dataset
    let dataset = MnistDataset::load("./data");
    let mut dataloader = DataLoader::new(&dataset, 64, true).iter();

    // Build model
    let mut model = Sequential::new(vec![
        Box::new(Conv2D::new(1, 32, 3, 3)),
        Box::new(ReLU::new()),
        Box::new(MaxPool2D::new(2, 2, 2, 2)),
        Box::new(Flatten::new()),
        Box::new(Dense::new(32 * 13 * 13, 128)),
        Box::new(ReLU::new()),
        Box::new(Dense::new(128, 10)),
    ]);

    // Optimizer
    let mut optimizer = SGD::new(0.01);

    // Training loop
    let epochs = 5;
    for epoch in 1..=epochs {
        println!("Epoch {}/{}", epoch, epochs);
        let mut total_loss = 0.0;
        let mut batches = 0;

        // Reset dataloader iterator each epoch
        for (images, labels) in DataLoader::new(&dataset, 64, true).iter() {
            // Forward
            let logits = model.forward(&images);
            let (loss, grad_loss) = softmax_cross_entropy(&logits, &labels);
            total_loss += loss;
            batches += 1;

            // Backward
            let _ = model.backward(&grad_loss);

            // Update
            model.update(&mut optimizer);
        }

        println!("  Avg Loss: {}", total_loss / batches as f32);
    }
}

