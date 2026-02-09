use burn::backend::{Autodiff, Wgpu};
use std::path::Path;

use burn::config::Config;
use lilith::{data, model, training};

fn main() {
    // Use WGPU backend (works on CPU/GPU)
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // Load training config
    let config = match model::TrainingConfig::load("training.json") {
        Ok(config) => config,
        Err(err) => {
            println!("Training config not found or invalid, using defaults: {err}");
            model::TrainingConfig::init()
        }
    };

    // Create model
    let model = model::ImagePreferenceModel::<MyAutodiffBackend>::new(&device);

    // Load datasets
    let train_dataset =
        data::load_dataset(Path::new("data/train_labels.csv"), Path::new("data/images"));

    let valid_dataset =
        data::load_dataset(Path::new("data/valid_labels.csv"), Path::new("data/images"));

    // Train
    let trained_model =
        training::train(model, device.clone(), train_dataset, valid_dataset, config);

    // Save model
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

    let recorder: burn::record::NamedMpkFileRecorder<FullPrecisionSettings> =
        NamedMpkFileRecorder::new();
    trained_model
        .save_file("model", &recorder)
        .expect("Failed to save model");

    println!("Training complete! Model saved to ./model");
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
