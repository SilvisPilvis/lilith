use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::LearningRate;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};

use crate::data::{ImageBatch, ImageBatcher, ImageItem};
use crate::model::{ImagePreferenceModel, TrainingConfig};

impl<B: AutodiffBackend> TrainStep for ImagePreferenceModel<B> {
    type Input = ImageBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward(batch.images);
        let target = batch.preferences;
        let loss = MseLoss::new().forward(output.clone(), target.clone(), Reduction::Mean);
        let grads = loss.clone().backward();
        let train_item = RegressionOutput::new(loss, output, target);
        TrainOutput::new(self, grads, train_item)
    }
}

impl<B: Backend> InferenceStep for ImagePreferenceModel<B> {
    type Input = ImageBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.images);
        let target = batch.preferences;
        let loss = MseLoss::new().forward(output.clone(), target.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, target)
    }
}

pub fn train<B: AutodiffBackend>(
    model: ImagePreferenceModel<B>,
    device: B::Device,
    train_dataset: impl burn::data::dataset::Dataset<ImageItem> + 'static,
    valid_dataset: impl burn::data::dataset::Dataset<ImageItem> + 'static,
    config: TrainingConfig,
) -> ImagePreferenceModel<B::InnerBackend> {
    let batcher_train = ImageBatcher::<B>::new(device.clone(), (224, 224));
    let batcher_valid = ImageBatcher::<B::InnerBackend>::new(device.clone(), (224, 224));

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let optim = config.optimizer.init();
    let lr_scheduler: LearningRate = config.learning_rate;

    let learner = Learner::new(model, optim, lr_scheduler);

    let result = SupervisedTraining::new(
        config.checkpoints.clone(),
        dataloader_train,
        dataloader_valid,
    )
    .num_epochs(config.num_epochs)
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .launch(learner);

    result.model
}
