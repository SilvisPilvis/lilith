use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    conv::Conv2d, conv::Conv2dConfig, modules::pool::AdaptiveAvgPool2d,
    modules::pool::AdaptiveAvgPool2dConfig, Linear, LinearConfig, Relu,
};
use burn::optim::AdamWConfig;
use burn::tensor::Tensor;
// use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ImagePreferenceModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
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
        }
    }
}

impl<B: Backend> ImagePreferenceModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let conv3 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let pool = AdaptiveAvgPool2dConfig::new([8, 8]).init();
        let fc1 = LinearConfig::new(128 * 8 * 8, 256).init(device);
        let fc2 = LinearConfig::new(256, 1).init(device);
        let activation = Relu::new();
        Self {
            conv1,
            conv2,
            conv3,
            pool,
            fc1,
            fc2,
            activation,
        }
    }

    // pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
    //     let mut x = self.conv1.forward(input);
    //     x = self.activation.forward(x);
    //     x = self.conv2.forward(x);
    //     x = self.activation.forward(x);
    //     x = self.conv3.forward(x);
    //     x = self.activation.forward(x);
    //     x = self.pool.forward(x);
    //     x = x.flatten(1, 3);
    //     x = self.fc1.forward(x);
    //     x = self.activation.forward(x);
    //     x = self.fc2.forward(x);
    //     sigmoid(x)
    // }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self.conv1.forward(input);
        x = self.activation.forward(x);
        x = self.conv2.forward(x);
        x = self.activation.forward(x);
        x = self.conv3.forward(x);
        x = self.activation.forward(x);

        let x = self.pool.forward(x); // still [B, C, H, W]
                                      // let x = x.reshape([x.dims()[0], 128 * 8 * 8]); // or x.flatten(1, 3)

        let x = x.flatten(1, 3);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);

        burn::tensor::activation::sigmoid(x)
    }
}
