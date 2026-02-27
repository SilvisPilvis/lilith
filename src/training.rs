use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::LearningRate;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::checkpoint::KeepLastNCheckpoints;
use burn::train::metric::LossMetric;
use burn::train::renderer::{tui::TuiMetricsRenderer, CliMetricsRenderer};
use burn::train::Interrupter;
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use std::io::IsTerminal;

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

    let interrupter = Interrupter::new();

    let use_tui = std::io::stdout().is_terminal();

    let mut training = SupervisedTraining::new(
        config.checkpoints.clone(),
        dataloader_train,
        dataloader_valid,
    )
    .with_interrupter(interrupter.clone());

    let recorder: NamedMpkFileRecorder<FullPrecisionSettings> = NamedMpkFileRecorder::new();
    training = training
        .with_file_checkpointer(recorder)
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(config.num_checkpoints));

    if use_tui {
        training = training.renderer(TuiMetricsRenderer::new(interrupter, config.checkpoint));
    } else {
        println!("TUI disabled (no TTY). Using CLI renderer.");
        training = training.renderer(CliMetricsRenderer::new());
    }

    if let Some(checkpoint) = config.checkpoint {
        training = training.checkpoint(checkpoint);
    }

    let result = training
        .num_epochs(config.num_epochs)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .summary()
        .launch(learner);

    result.model
}
