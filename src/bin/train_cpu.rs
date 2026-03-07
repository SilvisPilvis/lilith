use burn::backend::Autodiff;
use burn::config::Config;
use burn::data::dataset::Dataset;
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

    let train_labels_path = config.train_labels_path.clone();
    let valid_labels_path = config.valid_labels_path.clone();
    let images_dir = config.images_dir.clone();

    if config.auto_stratified_split {
        if train_labels_path != valid_labels_path {
            println!(
                "auto_stratified_split is enabled; ignoring valid_labels_path and splitting train_labels_path instead"
            );
        }

        let items = data::load_dataset_items(
            Path::new(&train_labels_path).to_path_buf(),
            Path::new(&images_dir).to_path_buf(),
        );
        let (train_split, valid_split) = data::stratified_split_dataset(
            items,
            config.valid_split_ratio,
            config.stratified_bins,
            config.stratified_split_seed,
        );

        println!(
            "Using stratified split from single CSV: {} train, {} valid",
            train_split.len(),
            valid_split.len()
        );

        let trained_model = training::train(model, device, train_split, valid_split, config);
        save_model(trained_model);
        println!("Training complete! Model saved to ./model");
        return;
    }

    let train_dataset = data::load_dataset(
        Path::new(&train_labels_path).to_path_buf(),
        Path::new(&images_dir).to_path_buf(),
    );
    let valid_dataset = data::load_dataset(
        Path::new(&valid_labels_path).to_path_buf(),
        Path::new(&images_dir).to_path_buf(),
    );

    println!(
        "Loaded train/valid datasets: {} train, {} valid",
        train_dataset.len(),
        valid_dataset.len()
    );

    let trained_model = training::train(model, device, train_dataset, valid_dataset, config);

    save_model(trained_model);
    println!("Training complete! Model saved to ./model");
}

fn save_model<B: Backend>(trained_model: model::ImagePreferenceModel<B>) {
    let recorder: burn::record::NamedMpkFileRecorder<FullPrecisionSettings> =
        NamedMpkFileRecorder::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Failed to save model");
}
