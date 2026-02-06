#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use std::path::Path;

fn main() {
    // Use WGPU backend (works on CPU/GPU)
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    // Create model
    let model = model::ImagePreferenceModel::<MyAutodiffBackend>::new(&device);

    // Load datasets
    let train_dataset =
        data::load_dataset(Path::new("data/train_labels.csv"), Path::new("data/images"));

    let valid_dataset =
        data::load_dataset(Path::new("data/valid_labels.csv"), Path::new("data/images"));

    // Train
    let trained_model = training::train(
        model,
        device.clone(),
        train_dataset,
        valid_dataset,
        50, // epochs
    );

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

    #[test]
    fn test_inference() {
        type Backend = burn::backend::Wgpu;
        let device = Default::default();

        let model = model::ImagePreferenceModel::<Backend>::new(&device);
        let predictor = inference::PreferencePredictor::new(model, device);

        // Load test image
        let img = image::open("test_image.jpg").unwrap();
        let score = predictor.predict(&img);

        println!("Preference score: {}", score);
        assert!((0.0..=1.0).contains(&score));
    }
}
