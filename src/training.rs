use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::HuberLossConfig;
use burn::nn::loss::Reduction;
use burn::optim::LearningRate;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::Interrupter;
use burn::train::checkpoint::KeepLastNCheckpoints;
use burn::train::metric::LossMetric;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Adaptor, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes,
    NumericEntry,
};
use burn::train::renderer::{CliMetricsRenderer, tui::TuiMetricsRenderer};
use burn::train::{
    InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
};
use std::io::IsTerminal;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::data::{ImageBatch, ImageBatcher, ImageItem};
use crate::model::{ImagePreferenceModel, TrainingConfig};

struct RegressionMetricsInput<B: Backend> {
    output: Tensor<B, 2>,
    targets: Tensor<B, 2>,
}

impl<B: Backend> RegressionMetricsInput<B> {
    fn new(output: Tensor<B, 2>, targets: Tensor<B, 2>) -> Self {
        Self { output, targets }
    }
}

impl<B: Backend> Adaptor<RegressionMetricsInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> RegressionMetricsInput<B> {
        RegressionMetricsInput::new(self.output.clone(), self.targets.clone())
    }
}

#[derive(Clone)]
struct MaeMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> MaeMetric<B> {
    fn new() -> Self {
        Self {
            name: Arc::new("MAE".to_string()),
            state: NumericMetricState::default(),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MaeMetric<B> {
    type Input = RegressionMetricsInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let [batch_size, _] = item.output.dims();
        let mae = item
            .output
            .clone()
            .sub(item.targets.clone())
            .abs()
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap();

        self.state.update(
            mae,
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for MaeMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Clone)]
struct SpearmanMetric<B: Backend> {
    name: MetricName,
    preds: Vec<f64>,
    targets: Vec<f64>,
    current: f64,
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> SpearmanMetric<B> {
    fn new() -> Self {
        Self {
            name: Arc::new("Spearman".to_string()),
            preds: Vec::new(),
            targets: Vec::new(),
            current: f64::NAN,
            state: NumericMetricState::default(),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for SpearmanMetric<B> {
    type Input = RegressionMetricsInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let preds = tensor_to_vec(&item.output);
        let targets = tensor_to_vec(&item.targets);
        let batch_size = preds.len();

        self.preds.extend(preds);
        self.targets.extend(targets);
        self.current = spearman_correlation(&self.preds, &self.targets);

        self.state.update(
            self.current,
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    fn clear(&mut self) {
        self.preds.clear();
        self.targets.clear();
        self.current = f64::NAN;
        self.state.reset();
    }
}

impl<B: Backend> Numeric for SpearmanMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

fn tensor_to_vec<B: Backend>(tensor: &Tensor<B, 2>) -> Vec<f64> {
    tensor.clone().into_data().iter::<f64>().collect()
}

fn spearman_correlation(preds: &[f64], targets: &[f64]) -> f64 {
    if preds.len() != targets.len() || preds.len() < 2 {
        return 0.0;
    }

    let pred_ranks = average_ranks(preds);
    let target_ranks = average_ranks(targets);
    pearson_correlation(&pred_ranks, &target_ranks)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;

    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }

        let avg_rank = (start + end - 1) as f64 / 2.0 + 1.0;
        for i in start..end {
            ranks[indexed[i].0] = avg_rank;
        }

        start = end;
    }

    ranks
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n != ys.len() || n < 2 {
        return 0.0;
    }

    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    let denominator = (denom_x * denom_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

impl<B: AutodiffBackend> TrainStep for ImagePreferenceModel<B> {
    type Input = ImageBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward(batch.images);
        let target = batch.preferences;
        let loss = HuberLossConfig::new(1.0).init().forward(
            output.clone(),
            target.clone(),
            Reduction::Mean,
        );
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
        let loss = HuberLossConfig::new(1.0).init().forward(
            output.clone(),
            target.clone(),
            Reduction::Mean,
        );
        RegressionOutput::new(loss, output, target)
    }
}

pub fn train<B: AutodiffBackend>(
    model: ImagePreferenceModel<B>,
    device: B::Device,
    // Burn's DataLoader builder requires owned datasets with a 'static lifetime.
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
        .metric_valid_numeric(MaeMetric::new())
        .metric_valid_numeric(SpearmanMetric::new())
        .summary()
        .launch(learner);

    result.model
}

pub fn save_model<B: Backend>(trained_model: ImagePreferenceModel<B>) {
    use burn::module::Module;

    let recorder: NamedMpkFileRecorder<FullPrecisionSettings> = NamedMpkFileRecorder::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Failed to save model");
}
