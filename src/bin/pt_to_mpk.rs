use anyhow::{Context, Result, bail};
use burn::backend::NdArray;
use burn::module::{Module, Param};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Shape, Tensor, TensorData};
use lilith::model::ImagePreferenceModel;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

type Backend = NdArray;

#[derive(Deserialize)]
#[serde(untagged)]
enum WeightValue {
    Scalar(f32),
    List(Vec<WeightValue>),
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .context("Usage: cargo run --bin pt_to_mpk -- <weights.json> [output_path]")?;
    let output = args.next().unwrap_or_else(|| "model_from_pt".to_string());

    let input_path = PathBuf::from(&input);
    let output_path = PathBuf::from(&output);

    if input_path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pt"))
    {
        bail!(
            "Direct .pt loading is not supported in this Rust binary. First convert the PyTorch checkpoint to JSON with convert.py, then run this tool on that JSON file."
        );
    }

    let weights = load_weights(&input_path)?;
    let device = Default::default();
    let mut model = ImagePreferenceModel::<Backend>::new(&device);

    let conv1_weight = tensor4(&weights, "conv1.weight", [32, 3, 3, 3], false, &device)?;
    let conv1_bias = tensor1(&weights, "conv1.bias", [32], &device)?;
    let conv2_weight = tensor4(&weights, "conv2.weight", [64, 32, 3, 3], false, &device)?;
    let conv2_bias = tensor1(&weights, "conv2.bias", [64], &device)?;
    let conv3_weight = tensor4(&weights, "conv3.weight", [128, 64, 3, 3], false, &device)?;
    let conv3_bias = tensor1(&weights, "conv3.bias", [128], &device)?;
    let fc1_weight = tensor2(&weights, "fc1.weight", [128, 128], true, &device)?;
    let fc1_bias = tensor1(&weights, "fc1.bias", [128], &device)?;
    let fc2_weight = tensor2(&weights, "fc2.weight", [1, 128], true, &device)?;
    let fc2_bias = tensor1(&weights, "fc2.bias", [1], &device)?;

    model.conv1.weight = replace_param(model.conv1.weight, conv1_weight);
    model.conv1.bias = Some(replace_param(
        model.conv1.bias.context("conv1 is missing bias param")?,
        conv1_bias,
    ));
    model.conv2.weight = replace_param(model.conv2.weight, conv2_weight);
    model.conv2.bias = Some(replace_param(
        model.conv2.bias.context("conv2 is missing bias param")?,
        conv2_bias,
    ));
    model.conv3.weight = replace_param(model.conv3.weight, conv3_weight);
    model.conv3.bias = Some(replace_param(
        model.conv3.bias.context("conv3 is missing bias param")?,
        conv3_bias,
    ));
    model.fc1.weight = replace_param(model.fc1.weight, fc1_weight);
    model.fc1.bias = Some(replace_param(
        model.fc1.bias.context("fc1 is missing bias param")?,
        fc1_bias,
    ));
    model.fc2.weight = replace_param(model.fc2.weight, fc2_weight);
    model.fc2.bias = Some(replace_param(
        model.fc2.bias.context("fc2 is missing bias param")?,
        fc2_bias,
    ));

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(output_path.clone(), &recorder)
        .with_context(|| format!("Failed to save MPK file to {}", output_path.display()))?;

    println!("Converted {} -> {}.mpk", input_path.display(), output_path.display());
    println!(
        "Note: linear layers are transposed during import because PyTorch stores them as [out, in] while Burn expects [in, out]."
    );

    Ok(())
}

fn load_weights(path: &Path) -> Result<HashMap<String, WeightValue>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read weights file {}", path.display()))?;
    let weights = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse JSON weights from {}", path.display()))?;
    Ok(weights)
}

fn tensor1(
    weights: &HashMap<String, WeightValue>,
    name: &str,
    shape: [usize; 1],
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Tensor<Backend, 1>> {
    let flat = flatten_weight(weights, name)?;
    expect_len(name, &flat, shape.iter().product())?;
    Ok(Tensor::from_data(TensorData::new(flat, Shape::new(shape)), device))
}

fn tensor2(
    weights: &HashMap<String, WeightValue>,
    name: &str,
    pytorch_shape: [usize; 2],
    transpose: bool,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Tensor<Backend, 2>> {
    let flat = flatten_weight(weights, name)?;
    expect_len(name, &flat, pytorch_shape.iter().product())?;

    let values = if transpose {
        transpose_2d(&flat, pytorch_shape[0], pytorch_shape[1])
    } else {
        flat
    };

    let burn_shape = if transpose {
        [pytorch_shape[1], pytorch_shape[0]]
    } else {
        pytorch_shape
    };

    Ok(Tensor::from_data(
        TensorData::new(values, Shape::new(burn_shape)),
        device,
    ))
}

fn tensor4(
    weights: &HashMap<String, WeightValue>,
    name: &str,
    shape: [usize; 4],
    _transpose: bool,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<Tensor<Backend, 4>> {
    let flat = flatten_weight(weights, name)?;
    expect_len(name, &flat, shape.iter().product())?;
    Ok(Tensor::from_data(TensorData::new(flat, Shape::new(shape)), device))
}

fn flatten_weight(weights: &HashMap<String, WeightValue>, name: &str) -> Result<Vec<f32>> {
    let value = weights
        .get(name)
        .with_context(|| format!("Missing tensor '{name}' in JSON weights"))?;
    let mut flat = Vec::new();
    flatten_value(value, &mut flat);
    Ok(flat)
}

fn flatten_value(value: &WeightValue, out: &mut Vec<f32>) {
    match value {
        WeightValue::Scalar(v) => out.push(*v),
        WeightValue::List(values) => {
            for value in values {
                flatten_value(value, out);
            }
        }
    }
}

fn expect_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        bail!(
            "Tensor '{name}' has {} values, expected {expected}",
            values.len()
        );
    }
    Ok(())
}

fn transpose_2d(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    transposed
}

fn replace_param<const D: usize>(param: Param<Tensor<Backend, D>>, tensor: Tensor<Backend, D>) -> Param<Tensor<Backend, D>> {
    let (id, _old_tensor, mapper) = param.consume();
    Param::from_mapped_value(id, tensor, mapper)
}
