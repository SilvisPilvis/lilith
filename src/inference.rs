use crate::model::ImagePreferenceModel;
use burn::{
    module::Module,
    prelude::ToElement,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend},
};
use image::DynamicImage;

pub struct PreferencePredictor<B: Backend> {
    model: ImagePreferenceModel<B>,
    device: B::Device,
    input_size: (u32, u32),
}

impl<B: Backend> PreferencePredictor<B> {
    pub fn new(model: ImagePreferenceModel<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            input_size: (224, 224),
        }
    }

    /// Load a saved model from file - replaces `new()` for loading saved weights
    pub fn from_file(path: &str, device: B::Device) -> Self
    where
        B: Backend, // Ensure B implements Backend (already in struct bound, but explicit here)
    {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        // Create empty model structure first, then load weights into it
        let model = ImagePreferenceModel::<B>::new(&device);

        // Load weights - returns Result, handle it
        let model = model
            .load_file(path, &recorder, &device)
            .expect(&format!("Failed to load model from: {}", path));

        Self::new(model, device)
    }

    /// Alternative: Try loading without panicking
    pub fn try_from_file(path: &str, device: B::Device) -> Result<Self, String>
    where
        B: Backend,
    {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = ImagePreferenceModel::<B>::new(&device);

        match model.load_file(path, &recorder, &device) {
            Ok(loaded_model) => Ok(Self::new(loaded_model, device)),
            Err(e) => Err(format!("Failed to load model from {}: {:?}", path, e)),
        }
    }

    /// Predict preference score (0.0 to 1.0) for any image
    pub fn predict(&self, image: &DynamicImage) -> f32 {
        // Preprocess
        let processed = crate::data::preprocess_image(image, self.input_size);
        let (w, h) = self.input_size;

        // Create 1D tensor from flat data, then reshape to 4D [1, 3, h, w]
        let tensor_1d = Tensor::<B, 1>::from_data(processed.as_slice(), &self.device);
        let tensor: Tensor<B, 4> = tensor_1d.reshape([1, 3, h as usize, w as usize]);

        // Forward pass
        let output = self.model.forward(tensor);

        // Extract scalar value - squeeze to 0D then convert
        output.squeeze::<0>().into_scalar().to_f32()
    }

    /// Batch prediction for multiple images
    pub fn predict_batch(&self, images: &[DynamicImage]) -> Vec<f32> {
        images.iter().map(|img| self.predict(img)).collect()
    }
}

