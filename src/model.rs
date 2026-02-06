use burn::{
    module::Module,
    nn::{
        LeakyRelu, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    optim::{AdamW, AdamWConfig},
    prelude::*,
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};

// A simple CNN that handles variable input sizes and outputs a preference score (0-1)
#[derive(Module, Debug)]
pub struct ImagePreferenceModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: LeakyRelu,
}

#[derive(Debug, Config)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    #[config(default = "checkpoints")] // String literal needs quotes in the attribute
    pub checkpoints: String,
}

impl<B: Backend> ImagePreferenceModel<B> {
    pub fn new(device: &B::Device) -> Self {
        // 3 input channels (RGB), progressively extract features
        let conv1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        // Adaptive pooling ensures fixed output size regardless of input dimensions
        let pool = AdaptiveAvgPool2dConfig::new([8, 8]).init();

        // Fully connected layers
        let fc1 = LinearConfig::new(128 * 8 * 8, 256).init(device);
        let fc2 = LinearConfig::new(256, 1).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            pool,
            fc1,
            fc2,
            activation: LeakyRelu {
                negative_slope: 0.2,
            },
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        // Pool to fixed size
        let x = self.pool.forward(x);

        let start_dim = x.clone().dims()[0];
        let end_dim: usize = 128 * 8 * 8;
        // Flatten
        // let x = x.reshape([x.clone().dims()[0], (128 * 8 * 8 as usize)]);
        let x = x.flatten(start_dim, end_dim);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);

        // Sigmoid to get 0-1 range
        sigmoid(x)
    }
}
