use crate::{data::ImageBatch, model::ImagePreferenceModel};
use burn::{
    config::Config, // Correct import for Config derive
    nn::loss::MseLoss,
    optim::{AdamW, AdamWConfig},
    tensor::{Tensor, backend::AutodiffBackend},
    train::{Learner, SupervisedTraining, TrainOutput, TrainStep, metric::LossMetric},
};

// Training step implementation
impl<B: AutodiffBackend> TrainStep<ImageBatch<B>, Tensor<B, 1>> for ImagePreferenceModel<B> {
    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<Tensor<B, 1>> {
        let output = self.forward(batch.images);
        let target = batch.preferences;
        let loss = MseLoss::new().forward(output.clone(), target);
        TrainOutput::new(self, loss, output)
    }
}

pub fn train<B: AutodiffBackend>(
    model: ImagePreferenceModel<B>,
    device: B::Device,
    train_dataset: impl burn::data::dataset::Dataset<crate::data::ImageItem> + 'static,
    valid_dataset: impl burn::data::dataset::Dataset<crate::data::ImageItem> + 'static,
    config: TrainingConfig,
) -> ImagePreferenceModel<B> {
    use burn::data::dataloader::DataLoaderBuilder;
    // Create batchers
    let batcher_train =
        crate::data::ImageBatcher::<B::InnerBackend>::new(device.clone(), (224, 224));
    let batcher_valid =
        crate::data::ImageBatcher::<B::InnerBackend>::new(device.clone(), (224, 224));

    // Build data loaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Initialize AdamW with learning rate as first parameter to init()
    let optim = AdamWConfig::new().init::<B, ImagePreferenceModel<B>>(); // lr passed to init()

    // Create supervised training
    let supervised_training =
        SupervisedTraining::new(&config.checkpoint_dir, dataloader_train, dataloader_valid)
            .num_epochs(config.num_epochs)
            .summary();

    // Launch training
    supervised_training.launch(Learner::new(model, optim, device))
}
