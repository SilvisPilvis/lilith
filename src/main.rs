#![recursion_limit = "256"]
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use std::path::Path;

use burn::config::Config;
use lilith::{data, model, training};

fn main() {
    // Use WGPU backend (works on CPU/GPU)
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    // let validated =
    //     data::validate_start_bytes("/home/silvestrs/Desktop/projects/lilith/data/images");

    // if !validated {
    //     println!("Validation failed");
    //     return;
    // }

    // Load training config
    let config = match model::TrainingConfig::load("training.json") {
        Ok(config) => config,
        Err(err) => {
            println!("Training config not found or invalid, using defaults: {err}");
            model::TrainingConfig::init()
        }
    };

    let device = parse_device(&config.device).unwrap_or_else(|| {
        println!(
            "Invalid device '{}', falling back to default",
            config.device
        );
        WgpuDevice::DefaultDevice
    });

    // Create model
    let model = model::ImagePreferenceModel::<MyAutodiffBackend>::new(&device);

    // Load datasets
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

        let trained_model =
            training::train(model, device.clone(), train_split, valid_split, config);

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

    // Train
    let trained_model =
        training::train(model, device.clone(), train_dataset, valid_dataset, config);

    save_model(trained_model);
    println!("Training complete! Model saved to ./model");
}

fn save_model<B: burn::tensor::backend::Backend>(trained_model: model::ImagePreferenceModel<B>) {
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

    let recorder: burn::record::NamedMpkFileRecorder<FullPrecisionSettings> =
        NamedMpkFileRecorder::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Failed to save model");
}

fn parse_device(value: &str) -> Option<WgpuDevice> {
    let value = value.trim().to_lowercase();

    if value == "default" || value == "auto" {
        return Some(WgpuDevice::DefaultDevice);
    }
    if value == "cpu" {
        return Some(WgpuDevice::Cpu);
    }

    if let Some(index) = value.strip_prefix("discrete:") {
        let index = index.parse::<usize>().ok()?;
        return Some(WgpuDevice::DiscreteGpu(index));
    }

    if let Some(index) = value.strip_prefix("integrated:") {
        let index = index.parse::<usize>().ok()?;
        return Some(WgpuDevice::IntegratedGpu(index));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use lilith::inference;

    #[test]
    fn test_inference() {
        type Backend = burn::backend::Wgpu;
        let device = Default::default();

        let model = model::ImagePreferenceModel::<Backend>::new(&device);
        let predictor = inference::PreferencePredictor::new(model, device);

        // Load test image (skip if missing)
        let img = match image::open("test_image.jpg") {
            Ok(image) => image,
            Err(image::ImageError::IoError(err)) if err.kind() == std::io::ErrorKind::NotFound => {
                println!("Skipping test: test_image.jpg not found");
                return;
            }
            Err(err) => panic!("Failed to open test_image.jpg: {err}"),
        };
        let score = predictor.predict(&img);

        println!("Preference score: {}", score);
        assert!((0.0..=1.0).contains(&score));
    }
}
