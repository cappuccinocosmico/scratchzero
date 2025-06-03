use crate::mnist::data::mnist::MnistDataset;
use crate::mnist::tensor::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// DataLoader for batching the dataset.
pub struct DataLoader<'a> {
    dataset: &'a MnistDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a MnistDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.labels.len()).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader { dataset, batch_size, shuffle, indices, pos: 0 }
    }

    pub fn iter(&mut self) -> DataLoaderIter<'a> {
        self.pos = 0;
        DataLoaderIter { loader: self }
    }
}

/// Iterator for DataLoader
pub struct DataLoaderIter<'a> {
    loader: &'a mut DataLoader<'a>,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let loader = &mut *self.loader;
        if loader.pos >= loader.indices.len() {
            return None;
        }
        let end = (loader.pos + loader.batch_size).min(loader.indices.len());
        let batch_idx = &loader.indices[loader.pos..end];
        let bs = batch_idx.len();
        let mut images = Tensor::zeros(&[bs, 1, 28, 28]);
        let mut labels = Tensor::zeros(&[bs, 10]);
        for (i, &idx) in batch_idx.iter().enumerate() {
            // copy image data
            let img_start = idx * 28 * 28;
            for j in 0..28*28 {
                images.data[i * 28 * 28 + j] = loader.dataset.images.data[img_start + j];
            }
            // one-hot label
            labels.data[i * 10 + loader.dataset.labels[idx] as usize] = 1.0;
        }
        loader.pos = end;
        Some((images, labels))
    }
}