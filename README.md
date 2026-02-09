# Lilith

Small Rust app that trains an image preference model using Burn v0.20 (WGPU backend) and runs inference on images.

## Requirements

- Rust toolchain (stable)
- A compatible GPU or CPU for WGPU

## Project Layout

- `data/train_labels.csv`
- `data/valid_labels.csv`
- `data/images/`
- `test_image.jpg` (optional, for the unit test)
- `training.json`
- `inference.json` (optional, for inference config)

CSV format (headers required) for both `data/train_labels.csv` and `data/valid_labels.csv`:

```
image_path,preference
example.jpg,0.85
another.png,0.12
```

`image_path` is resolved relative to `data/images/`.
Each row is a single image with a preference score from 0.0 to 1.0.

`data/train_labels.csv` is used for training updates, while `data/valid_labels.csv` is held out for validation metrics during training.
Recommended split: 80/20 (train/valid) for smaller datasets, or 90/10 if data is limited.

The image paths in valid_labels should be different from train_labels so that you can test if the model infers correctly.

## Run

Train and save the model:

```
cargo run
```

Training reads hyperparameters from `training.json` (or falls back to defaults) and writes the model to `./model`.

Run tests:

```
cargo test
```

The inference test is skipped if `test_image.jpg` is missing.

Run inference with the separate binary:

```
cargo run --bin infer -- model path/to/image.jpg
```

Run inference with a custom config:

```
cargo run --bin infer -- --config inference.json
```

## Training Config

Training defaults live in `src/model.rs` under `TrainingConfig::init()`:

- `num_epochs`: total epochs to train
- `batch_size`: samples per batch
- `num_workers`: dataloader worker threads
- `seed`: RNG seed for shuffling
- `learning_rate`: optimizer learning rate
- `checkpoints`: directory for training artifacts

These defaults are mirrored in `training.json`. To change them, edit `training.json`.

Example `training.json` (comments shown for clarity; remove `//` lines before use):

```json
{
  "optimizer": {
    "beta_1": 0.9,            // AdamW beta_1
    "beta_2": 0.999,          // AdamW beta_2
    "epsilon": 0.00001,       // AdamW epsilon
    "weight_decay": 0.0001,   // AdamW weight decay
    "cautious_weight_decay": false,
    "grad_clipping": null     // optional gradient clipping config
  },
  "num_epochs": 10,           // total epochs
  "batch_size": 32,           // batch size
  "num_workers": 4,           // dataloader workers
  "seed": 42,                 // shuffle seed
  "learning_rate": 0.0001,    // learning rate
  "checkpoints": "checkpoints" // output directory
}
```

Example `inference.json` (comments shown for clarity; remove `//` lines before use):

```json
{
  "model_path": "model",          // path to saved model
  "image_path": "path/to/image.jpg" // image to score
}
```

## GPU Setup

This project uses Burn's WGPU backend. By default it will select a suitable device automatically.
If you need to force a specific graphics API (e.g., Vulkan), initialize the WGPU runtime before creating the device.

Example (in `main`):

```rust
use burn::backend::wgpu;

let device = wgpu::WgpuDevice::default();
wgpu::init_setup::<wgpu::graphics::Vulkan>(&device, Default::default());
```

## Inference

The `PreferencePredictor` in `src/inference.rs` can run predictions on images.
After training, load the saved model and call `predict`:

```rust
use burn::backend::Wgpu;
use lilith::inference::PreferencePredictor;

let device = Default::default();
let predictor = PreferencePredictor::<Wgpu>::from_file("model", device);
let img = image::open("path/to/image.jpg").unwrap();
let score = predictor.predict(&img);
println!("score: {score}");
```
