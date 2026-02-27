use burn::backend::Autodiff;
use burn::config::Config;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use std::path::Path;

use lilith::{data, model, training};

fn main() {
    type MyBackend = burn::backend::NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = <MyBackend as Backend>::Device::default();

    let config = match model::TrainingConfig::load("training.json") {
        Ok(config) => config,
        Err(err) => {
            println!("Training config not found or invalid, using defaults: {err}");
            model::TrainingConfig::init()
        }
    };

    let model = model::ImagePreferenceModel::<MyAutodiffBackend>::new(&device);

    let train_dataset =
        data::load_dataset(Path::new("data/train_labels.csv"), Path::new("data/images"));
    let valid_dataset =
        data::load_dataset(Path::new("data/valid_labels.csv"), Path::new("data/images"));

    let trained_model = training::train(model, device, train_dataset, valid_dataset, config);

    let recorder: burn::record::NamedMpkFileRecorder<FullPrecisionSettings> =
        NamedMpkFileRecorder::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Failed to save model");

    println!("Training complete! Model saved to ./model");
}
