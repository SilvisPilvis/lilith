use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, conv::Conv2d, conv::Conv2dConfig,
    modules::pool::AdaptiveAvgPool2d, modules::pool::AdaptiveAvgPool2dConfig, norm::BatchNorm,
    norm::BatchNormConfig,
};
use burn::optim::AdamWConfig;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ImagePreferenceModel<B: Backend> {
    pub block1: ResidualBlock<B>,
    pub block2: ResidualBlock<B>,
    pub block3: ResidualBlock<B>,
    pub block4: ResidualBlock<B>,
    pub global_pool: AdaptiveAvgPool2d,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub fc3: Linear<B>,
    pub dropout: Dropout,
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    pub conv1: Conv2d<B>,
    pub bn1: BatchNorm<B>,
    pub conv2: Conv2d<B>,
    pub bn2: BatchNorm<B>,
    pub skip_conv: Option<Conv2d<B>>,
    pub skip_bn: Option<BatchNorm<B>>,
    pub se_pool: AdaptiveAvgPool2d,
    pub se_reduce: Linear<B>,
    pub se_expand: Linear<B>,
}

impl<B: Backend> ResidualBlock<B> {
    fn new(
        device: &B::Device,
        in_channels: usize,
        out_channels: usize,
        stride: [usize; 2],
        use_dilation: bool,
    ) -> Self {
        let dilation = if use_dilation { [2, 2] } else { [1, 1] };
        let padding = if use_dilation { [2, 2] } else { [1, 1] };

        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
            .with_dilation(dilation)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let (skip_conv, skip_bn) = if in_channels != out_channels || stride != [1, 1] {
            let skip_conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride(stride)
                .init(device);
            let skip_bn = BatchNormConfig::new(out_channels).init(device);
            (Some(skip_conv), Some(skip_bn))
        } else {
            (None, None)
        };

        let se_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let se_hidden = (out_channels / 8).max(8);
        let se_reduce = LinearConfig::new(out_channels, se_hidden).init(device);
        let se_expand = LinearConfig::new(se_hidden, out_channels).init(device);

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            skip_conv,
            skip_bn,
            se_pool,
            se_reduce,
            se_expand,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match (&self.skip_conv, &self.skip_bn) {
            (Some(skip_conv), Some(skip_bn)) => skip_bn.forward(skip_conv.forward(input.clone())),
            _ => input.clone(),
        };

        let mut x = self.conv1.forward(input);
        x = self.bn1.forward(x);
        x = burn::tensor::activation::silu(x);

        x = self.conv2.forward(x);
        x = self.bn2.forward(x);

        let [batch, channels, _, _] = x.dims();
        let mut se: Tensor<B, 2> = self.se_pool.forward(x.clone()).flatten(1, 3);
        se = self.se_reduce.forward(se);
        se = burn::tensor::activation::silu(se);
        se = self.se_expand.forward(se);
        se = burn::tensor::activation::sigmoid(se);
        let se = se.reshape([batch, channels, 1, 1]);

        x = x.mul(se);
        x = x.add(residual);
        burn::tensor::activation::silu(x)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub checkpoints: String,
    pub device: String,
    pub checkpoint: Option<usize>,
    pub num_checkpoints: usize,
    pub train_labels_path: String,
    pub valid_labels_path: String,
    pub images_dir: String,
    pub auto_stratified_split: bool,
    pub valid_split_ratio: f32,
    pub stratified_bins: usize,
    pub stratified_split_seed: u64,
}

impl TrainingConfig {
    pub fn init() -> Self {
        Self {
            optimizer: AdamWConfig::new(),
            num_epochs: 10,
            batch_size: 32,
            num_workers: 4,
            seed: 42,
            learning_rate: 1e-4,
            checkpoints: "checkpoints".to_string(),
            device: "default".to_string(),
            checkpoint: None,
            num_checkpoints: 2,
            train_labels_path: "data/train_labels.csv".to_string(),
            valid_labels_path: "data/valid_labels.csv".to_string(),
            images_dir: "data/images".to_string(),
            auto_stratified_split: false,
            valid_split_ratio: 0.2,
            stratified_bins: 10,
            stratified_split_seed: 42,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        if self.num_epochs == 0 {
            return Err("num_epochs must be greater than 0".to_string());
        }
        if self.valid_split_ratio <= 0.0 || self.valid_split_ratio >= 1.0 {
            return Err("valid_split_ratio must be in (0.0, 1.0)".to_string());
        }
        if self.stratified_bins == 0 {
            return Err("stratified_bins must be greater than 0".to_string());
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err("learning_rate must be finite and greater than 0".to_string());
        }

        Ok(())
    }
}

impl<B: Backend> ImagePreferenceModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let block1 = ResidualBlock::new(device, 3, 32, [2, 2], false);
        let block2 = ResidualBlock::new(device, 32, 64, [2, 2], false);
        let block3 = ResidualBlock::new(device, 64, 96, [2, 2], true);
        let block4 = ResidualBlock::new(device, 96, 128, [2, 2], true);

        let global_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let fc1 = LinearConfig::new(128, 256).init(device);
        let fc2 = LinearConfig::new(256, 64).init(device);
        let fc3 = LinearConfig::new(64, 1).init(device);
        let dropout = DropoutConfig::new(0.3).init();

        Self {
            block1,
            block2,
            block3,
            block4,
            global_pool,
            fc1,
            fc2,
            fc3,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.block1.forward(input);
        x = self.block2.forward(x);
        x = self.block3.forward(x);
        x = self.block4.forward(x);

        let x = self.global_pool.forward(x).flatten(1, 3);

        // Use inline activations to keep the module simpler and avoid storing an extra activation field.
        let x = burn::tensor::activation::silu(self.fc1.forward(x));
        let x = self.dropout.forward(x);
        let x = burn::tensor::activation::silu(self.fc2.forward(x));
        let x = self.dropout.forward(x);
        let x = self.fc3.forward(x);

        burn::tensor::activation::sigmoid(x)
    }
}
