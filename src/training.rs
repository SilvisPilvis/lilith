use burn::data::dataloader::{DataLoaderBuilder, batcher::Batcher};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::train::{
    InferenceStep, Learner, MetricEntry, TrainOutput, TrainStep, metric::LossMetric,
};

use crate::data::{ImageBatch, ImageBatcher};
use crate::model::{ImagePreferenceModel, TrainingConfig};

impl<B: Backend> TrainStep<ImageBatch<B>, Tensor<B, 2>> for ImagePreferenceModel<B> {
    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<Tensor<B, 2>> {
        let output = self.forward(batch.images);
        let target = batch.preferences;
        let loss = burn::nn::loss::MseLoss::new().forward(output.clone(), target, None);
        let grads = loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> InferenceStep<ImageBatch<B>, Tensor<B, 2>> for ImagePreferenceModel<B> {
    fn step(&self, batch: ImageBatch<B>) -> Tensor<B, 2> {
        self.forward(batch.images)
    }
}

pub fn train<B: Backend>(
    model: ImagePreferenceModel<B>,
    device: B::Device,
    train_dataset: impl burn::data::dataset::Dataset<(String, f32)> + 'static,
    valid_dataset: impl burn::data::dataset::Dataset<(String, f32)> + 'static,
    epochs: usize,
) -> ImagePreferenceModel<B> {
    let mut config = TrainingConfig::init();
    config.num_epochs = epochs;
    let batcher_train = ImageBatcher::<B>::new(device.clone(), (224, 224));
    let batcher_valid = ImageBatcher::<B>::new(device.clone(), (224, 224));

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Box::new(train_dataset));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(Box::new(valid_dataset));

    let mut optim = config
        .optimizer
        .with_learning_rate(config.learning_rate)
        .init();

    let learner = Learner::new(&config.checkpoints)
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .metric(LossMetric::new())
        .build(model, optim);

    learner.fit(dataloader_train, dataloader_valid)
}
