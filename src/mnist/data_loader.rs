use crate::mnist::tensor::Tensor;
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::error::Error;

/// MNIST DataLoader for images and labels
pub struct DataLoader {
    pub images: Vec<Tensor<3>>, // shape [1, 28, 28]
    pub labels: Vec<usize>,
    pub batch_size: usize,
    index: usize,
}

impl DataLoader {
    /// Create a new DataLoader from MNIST idx files
    pub fn new(
        image_path: &str,
        label_path: &str,
        batch_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let images = read_mnist_images(image_path)?;
        let labels = read_mnist_labels(label_path)?;
        if images.len() != labels.len() {
            return Err("Number of images and labels do not match".into());
        }
        Ok(DataLoader { images, labels, batch_size, index: 0 })
    }

    /// Get next batch. Returns None when no more data.
    pub fn next_batch(&mut self) -> Option<(Vec<Tensor<3>>, Vec<usize>)> {
        if self.index >= self.images.len() {
            return None;
        }
        let end = usize::min(self.index + self.batch_size, self.images.len());
        let batch_imgs = self.images[self.index..end].to_vec();
        let batch_lbls = self.labels[self.index..end].to_vec();
        self.index = end;
        Some((batch_imgs, batch_lbls))
    }

    /// Reset the loader to start
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

fn read_u32_be(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_mnist_images(path: &str) -> Result<Vec<Tensor<3>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let magic = read_u32_be(&mut reader)?;
    if magic != 2051 {
        return Err(format!("Invalid magic number for images: {}", magic).into());
    }
    let count = read_u32_be(&mut reader)? as usize;
    let rows = read_u32_be(&mut reader)? as usize;
    let cols = read_u32_be(&mut reader)? as usize;
    let mut images = Vec::with_capacity(count);
    let mut buf = vec![0u8; rows * cols];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        // convert to Tensor<3> with shape [1, rows, cols]
        let mut data = Vec::with_capacity(rows * cols);
        for &b in buf.iter() {
            data.push((b as f32) / 255.0);
        }
        let tensor = Tensor::from_vec_unchecked(data, [1, rows, cols]);
        images.push(tensor);
    }
    Ok(images)
}

fn read_mnist_labels(path: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let magic = read_u32_be(&mut reader)?;
    if magic != 2049 {
        return Err(format!("Invalid magic number for labels: {}", magic).into());
    }
    let count = read_u32_be(&mut reader)? as usize;
    let mut labels = Vec::with_capacity(count);
    let mut buf = [0u8; 1];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        labels.push(buf[0] as usize);
    }
    Ok(labels)
}