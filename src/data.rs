use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, InMemDataset},
};
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ImageItem {
    pub image_path: String,
    pub preference: f32, // 0.0 to 1.0
}

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub preferences: Tensor<B, 2>,
}

pub struct ImageBatcher<B: Backend> {
    device: B::Device,
    target_size: (u32, u32), // (width, height)
}

impl<B: Backend> ImageBatcher<B> {
    pub fn new(device: B::Device, target_size: (u32, u32)) -> Self {
        Self {
            device,
            target_size,
        }
    }
}

impl<B: Backend> Batcher<B, ImageItem, ImageBatch<B>> for ImageBatcher<B> {
    fn batch(&self, items: Vec<ImageItem>, device: &B::Device) -> ImageBatch<B> {
        let batch_size = items.len();
        let (width, height) = self.target_size;

        // Stack all images into a batch
        let mut image_data = Vec::with_capacity(batch_size * 3 * height as usize * width as usize);
        let mut preferences = Vec::with_capacity(batch_size);

        for item in items {
            // Load and preprocess image
            let img = image::open(&item.image_path).expect("Failed to load image");
            let processed = preprocess_image(&img, self.target_size);

            // processed is Vec<f32> with shape [3, H, W] (CHW format)
            image_data.extend(processed);
            preferences.push(item.preference);
        }

        // Convert to tensors
        // Shape: [batch_size, channels, height, width]
        let image_tensor = Tensor::from_data(
            TensorData::new(
                image_data,
                Shape::new([batch_size, 3, height as usize, width as usize]),
            ),
            device,
        );

        let pref_tensor = Tensor::from_data(
            TensorData::new(preferences, Shape::new([batch_size, 1])),
            device,
        );

        ImageBatch {
            images: image_tensor,
            preferences: pref_tensor,
        }
    }
}

/// Preprocess image: resize maintaining aspect ratio with letterboxing, normalize to 0-1
pub fn preprocess_image(img: &DynamicImage, target_size: (u32, u32)) -> Vec<f32> {
    let (target_w, target_h) = target_size;
    let (orig_w, orig_h) = img.dimensions();

    // Calculate scaling to fit within target while maintaining aspect ratio
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale) as u32;
    let new_h = (orig_h as f32 * scale) as u32;

    // Resize
    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

    // Create blank canvas (gray background) and paste resized image centered
    let mut canvas = image::RgbImage::new(target_w, target_h);

    // Fill with gray (0.5 * 255)
    for pixel in canvas.pixels_mut() {
        *pixel = image::Rgb([128, 128, 128]);
    }

    let x_offset = (target_w - new_w) / 2;
    let y_offset = (target_h - new_h) / 2;

    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset as i64,
        y_offset as i64,
    );

    // Convert to CHW format and normalize to 0-1
    let mut data = Vec::with_capacity((3 * target_w * target_h) as usize);

    for c in 0..3 {
        for y in 0..target_h {
            for x in 0..target_w {
                let pixel = canvas.get_pixel(x, y);
                let val = pixel[c] as f32 / 255.0;
                data.push(val);
            }
        }
    }

    data
}

#[derive(Debug, Deserialize)]
struct CsvRecord {
    image_path: String,
    preference: f32,
}

pub fn load_dataset(csv_path: &Path, image_dir: &Path) -> impl Dataset<ImageItem> {
    let mut items = Vec::new();
    let mut reader = csv::Reader::from_path(csv_path).expect("Failed to read CSV");

    for result in reader.deserialize() {
        let record: CsvRecord = result.expect("Failed to parse record");
        let full_path = image_dir.join(&record.image_path);

        items.push(ImageItem {
            image_path: full_path.to_string_lossy().to_string(),
            preference: record.preference.clamp(0.0, 1.0),
        });
    }

    InMemDataset::new(items)
}
