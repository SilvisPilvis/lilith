use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::config::Config;
use lilith::inference::PreferencePredictor;
use std::env;
use std::process;

#[derive(Config, Debug)]
struct InferenceConfig {
    model_path: String,
    image_path: String,
    device: String,
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model-path> <image-path>", args[0]);
        eprintln!("   or: {} --config <inference.json>", args[0]);
        process::exit(1);
    }

    let (model_path, image_path, device) = if args.len() == 3 && args[1] == "--config" {
        let config = InferenceConfig::load(&args[2]).unwrap_or_else(|err| {
            eprintln!("Failed to load inference config {}: {}", args[2], err);
            process::exit(1);
        });
        (config.model_path, config.image_path, config.device)
    } else if args.len() == 3 {
        (args[1].clone(), args[2].clone(), "default".to_string())
    } else {
        eprintln!("Usage: {} <model-path> <image-path>", args[0]);
        eprintln!("   or: {} --config <inference.json>", args[0]);
        process::exit(1);
    };

    let device = parse_device(&device).unwrap_or_else(|| {
        eprintln!("Invalid device '{}', falling back to default", device);
        WgpuDevice::DefaultDevice
    });
    let predictor = PreferencePredictor::<Wgpu>::from_file(&model_path, device);

    let img = image::open(&image_path).unwrap_or_else(|err| {
        eprintln!("Failed to open image {}: {}", image_path, err);
        process::exit(1);
    });

    let score = predictor.predict(&img);
    println!("{score}");
}
