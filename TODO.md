# Lilith Model Improvement TODO

Current baseline: 3-layer CNN with true global average pooling to 1x1, sigmoid output, SmoothL1/Huber-style loss, MAE + Spearman validation (src/model.rs, src/training.rs)

## High Priority / High Impact

### 1. Pretrained Backbone
- Replace custom CNN with pretrained image encoder (ResNet/EfficientNet style)
- Freeze backbone initially, train only regression head
- Optionally fine-tune upper layers later
- **Expected impact**: Largest single improvement for aesthetic scoring

### 2. Better Loss Function
- Switch from MSE to SmoothL1/Huber loss
- Handles noisy subjective labels better than MSE
- File: `src/training.rs:25,39`
- **Expected impact**: More robust to outlier labels

### 3. Validation Metrics
- Add Spearman correlation as primary metric
- Add MAE alongside existing loss
- For preference ranking, Spearman > raw MSE
- File: `src/training.rs:97-101`
- **Expected impact**: Better signal for model selection

### 4. Stratified Data Splits
- Split train/valid with stratification by score bins (0.0-0.1, 0.1-0.2, etc.)
- Ensures all score ranges represented in both sets
- Implemented in `src/data.rs`; controlled by config in `training.json`
- **Expected impact**: More reliable validation

### 5. Input Normalization
- Normalize images to ImageNet mean/std if using pretrained backbone
- Current: only 0-1 scaling (src/data.rs:176)
- **Expected impact**: Better transfer learning

## Medium Priority

### 6. Training Augmentation
- Random crop/resize
- Horizontal flip (if semantically valid for aesthetics)
- Mild color jitter
- Optional slight rotation
- File: `src/data.rs` (in batcher)
- **Expected impact**: Better generalization

### 7. Global Average Pooling to 1x1
- Change AdaptiveAvgPool2d from [8,8] to [1,1]
- Reduces head parameters: 128*8*8=8192 -> 128
- Implemented in `src/model.rs`
- **Expected impact**: Simpler model, stronger channel features

### 8. K-Fold Cross-Validation
- Implement if dataset is small/medium
- Use for model selection, not every training run
- **Expected impact**: More reliable architecture/hyperparameter decisions

### 9. Learning Rate Schedule
- Add cosine decay or step LR
- Current: constant LR (src/training.rs:66)
- **Expected impact**: Better final convergence

### 10. Gradient Clipping
- Add to optimizer config for stability
- File: `src/model.rs` TrainingConfig
- **Expected impact**: More stable training

## Lower Priority / Later Experiments

### 11. Test-Time Augmentation (TTA)
- Average predictions over multiple augmented views
- Try only after baseline is solid
- **Expected impact**: Small-to-moderate boost, added inference cost

### 12. Squeeze-and-Excitation Blocks
- Add channel attention to backbone
- Lower risk than full attention mechanisms
- File: `src/model.rs`
- **Expected impact**: Mild improvement if backbone is custom

### 13. Sigmoid Alternatives
- Consider unconstrained output, clip only for display
- Sigmoid can saturate near 0/1
- **Expected impact**: Potentially easier optimization

### 14. Pairwise Ranking Loss
- If labels come from A vs B comparisons
- May outperform regression for preference tasks
- Requires data format change
- **Expected impact**: Could be significant if labels are ordinal

## Not Recommended

### Label Smoothing
- Designed for classification, not regression
- For noisy regression labels: prefer SmoothL1, target averaging, or ranking loss

### Perfectly Uniform Label Distribution
- Avoid distorting true data distribution
- Prefer stratified splits and mild reweighting

## Implementation Order

1. Add training augmentations
2. Pretrained backbone + normalization
3. K-fold CV for model selection
4. LR schedule + gradient clipping
5. TTA and SE blocks as final polish

## Current Architecture Summary

```
Input: [B, 3, 224, 224]
Conv1: 3 -> 32, 3x3, same padding, SiLU
Conv2: 32 -> 64, 3x3, same padding, SiLU
Conv3: 64 -> 128, 3x3, same padding, SiLU
Pool: AdaptiveAvgPool2d to [1, 1]
Flatten: 128
FC1: 128 -> 128, SiLU
FC2: 128 -> 1
Sigmoid -> [0, 1]

Loss: HuberLoss
Optimizer: AdamW (lr=1e-4)
Metrics: Loss, MAE, Spearman
```

## Target Architecture (After Upgrades)

```
Input: [B, 3, 224, 224]
Pretrained backbone (frozen initially)
GlobalAvgPool -> [B, features]
FC1: features -> 128, SiLU/Dropout
FC2: 128 -> 1
Sigmoid or unconstrained

Loss: SmoothL1Loss
Optimizer: AdamW + LR schedule
Metrics: Loss, MAE, Spearman
Augmentation: crop, flip, color jitter
```

Switch to pretrained:

MobileNetV3-Small
MobileNetV3-Large
EfficientNet-B0
EfficientNet-B1
ResNet-18
ResNet-34
