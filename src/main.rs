use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
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
    let train_dataset = data::load_dataset(
        Path::new("/home/silvestrs/Desktop/projects/lilith/data/train_labels.csv"),
        Path::new("/home/silvestrs/Desktop/projects/lilith/data/images"),
    );

    let valid_dataset = data::load_dataset(
        Path::new("/home/silvestrs/Desktop/projects/lilith/data/valid_labels.csv"),
        Path::new("/home/silvestrs/Desktop/projects/lilith/data/images"),
    );

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
