use burn::backend::NdArray;
use burn::config::Config;
use lilith::inference::PreferencePredictor;
use std::env;
use std::process;

#[derive(Config, Debug)]
struct InferenceConfig {
    model_path: String,
    image_path: String,
    device: Option<String>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model-path> <image-path>", args[0]);
        eprintln!("   or: {} --config <inference.json>", args[0]);
        process::exit(1);
    }

    let (model_path, image_path, device_hint) = if args.len() == 3 && args[1] == "--config" {
        let config = InferenceConfig::load(&args[2]).unwrap_or_else(|err| {
            eprintln!("Failed to load inference config {}: {}", args[2], err);
            process::exit(1);
        });
        (config.model_path, config.image_path, config.device)
    } else if args.len() == 3 {
        (args[1].clone(), args[2].clone(), None)
    } else {
        eprintln!("Usage: {} <model-path> <image-path>", args[0]);
        eprintln!("   or: {} --config <inference.json>", args[0]);
        process::exit(1);
    };

    if let Some(device) = device_hint {
        let device = device.trim().to_lowercase();
        if device != "cpu" && device != "default" && device != "auto" {
            eprintln!("infer_cpu ignores device '{}', using CPU", device);
        }
    }

    let device = Default::default();
    let predictor = PreferencePredictor::<NdArray>::from_file(&model_path, device);

    let img = image::open(&image_path).unwrap_or_else(|err| {
        eprintln!("Failed to open image {}: {}", image_path, err);
        process::exit(1);
    });

    let score = predictor.predict(&img);
    println!("{score}");
}
