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
- `device`: WGPU device selector (`default`, `cpu`, `discrete:0`, `integrated:0`)
- `checkpoint`: resume epoch (null to start fresh)
- `num_checkpoints`: number of checkpoints to keep (saved every epoch)
- `train_labels_path`: path to training CSV
- `valid_labels_path`: path to validation CSV
- `images_dir`: directory containing all images referenced by CSVs
- `auto_stratified_split`: if true, ignore `valid_labels_path` and create a stratified split from `train_labels_path`
- `valid_split_ratio`: validation fraction to use when `auto_stratified_split` is enabled
- `stratified_bins`: number of score bins used for stratified splitting
- `stratified_split_seed`: RNG seed for the stratified split shuffle

These defaults are mirrored in `training.json`. To change them, edit `training.json`.

If the training failed due to VRAM limitations, you can reduce the `batch_size` or `image_size` increase the `num_workers`.

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
  "checkpoints": "checkpoints", // output directory
  "device": "default",         // default | cpu | discrete:0 | integrated:0
  "checkpoint": null,          // resume epoch or null
  "num_checkpoints": 2,        // number of checkpoints to keep
  "train_labels_path": "data/train_labels.csv",
  "valid_labels_path": "data/valid_labels.csv",
  "images_dir": "data/images",
  "auto_stratified_split": false, // if true, split train_labels_path into train/valid
  "valid_split_ratio": 0.2,       // used only when auto_stratified_split=true
  "stratified_bins": 10,          // label bins for stratified split
  "stratified_split_seed": 42     // stratified split shuffle seed
}
```

If `auto_stratified_split` is enabled, the app builds both train and validation sets from `train_labels_path` and ignores `valid_labels_path`. This is safer and more explicit than inferring split behavior from matching file paths.

## Training Metrics

The Python training script reports several validation metrics for this regression task:

- `Val Loss`: the SmoothL1/Huber-style training loss on the validation set. Lower is better.
- `Val MAE`: mean absolute error between predicted score and target score. Lower is better.
- `Val Spearman`: rank correlation between predictions and labels. This answers "if I sort images by predicted score, does that order match my real preference order?" Higher is better.
- `BucketAcc`: both prediction and label are placed into coarse score buckets like `0.0-0.1`, `0.1-0.2`, and so on. This answers "did the model land in the right rough score range?" Higher is better.
- `Acc@0.05`, `Acc@0.10`, `Acc@0.20`: tolerance accuracies. For example, `Acc@0.10` is the fraction of predictions within `0.10` of the true score. Higher is better.

### What good values look like

- `Spearman`
  - Perfect: `1.0`
  - Very strong: `0.8+`
  - Useful: `0.5-0.8`
  - Weak: `0.2-0.5`
  - Bad: near `0.0` or negative

- `Acc@0.05`, `Acc@0.10`, `Acc@0.20`
  - Perfect: `1.0`
  - Higher is always better
  - `Acc@0.05` is strict, so it will usually be the lowest
  - `Acc@0.20` is forgiving, so it will usually be the highest
  - A healthy model should usually satisfy:
    - `Acc@0.20 >= Acc@0.10 >= Acc@0.05`

- `BucketAcc`
  - Perfect: `1.0`
  - With 10 equal score buckets, random guessing would be around `0.10` if the data were perfectly balanced
  - In practice, a good model should be clearly above random and ideally move toward `0.5+`, though the exact number depends on label distribution and task difficulty

### Which metrics matter most

- If you care most about ranking images by your taste, focus on `Spearman`
- If you care most about matching the exact score you would give, focus on `MAE` and `Acc@0.10`
- `BucketAcc` is best used as a rough sanity-check metric, not the main decision metric

Example `inference.json` (comments shown for clarity; remove `//` lines before use):

```json
{
  "model_path": "model",          // path to saved model
  "image_path": "path/to/image.jpg", // image to score
  "device": "default"             // default | cpu | discrete:0 | integrated:0
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
