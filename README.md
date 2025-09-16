# 🐾 AI-Based Pet Breed Recognition System

An intelligent pet breed detection system that utilizes **Vision Transformer (ViT)** to identify cat and dog breeds with high accuracy. This project combines multiple datasets and implements state-of-the-art deep learning techniques for robust breed classification.

## 📋 Table of Contents

- [Features](#-features)
- [Datasets](#-datasets)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training Process](#-training-process)
- [Evaluation](#-evaluation)
- [Prediction](#-prediction)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🐶 **Multi-breed Classification**: Supports both cat and dog breed recognition
- 🧠 **Vision Transformer (ViT)**: Uses Google's ViT-Base-Patch16-224 model
- 📊 **Class Balancing**: Automatic class distribution equalization
- 🔄 **Data Augmentation**: Built-in data preprocessing and augmentation
- 📈 **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score
- 💾 **Model Persistence**: Save and load trained models for inference
- 🖼️ **Easy Prediction**: Simple interface for breed prediction on new images

## 📊 Datasets

This project utilizes two high-quality datasets:

### 1. Stanford Dogs Dataset
- **Source**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Classes**: 120 dog breeds
- **Images**: ~20,580 high-quality dog images

### 2. Oxford-IIIT Pet Dataset
- **Source**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Classes**: 37 pet categories (cats and dogs)
- **Images**: ~7,349 pet images

### Combined Dataset Statistics
- **Total Classes**: 142 unique breeds
- **Total Images**: ~27,929 images
- **Train/Validation Split**: 80/20 ratio
- **Class Balancing**: Equalized to 200 samples per class (where available)

## 🧠 Model Architecture

### Vision Transformer (ViT)
- **Base Model**: `google/vit-base-patch16-224-in21k`
- **Input Size**: 224×224 pixels
- **Patch Size**: 16×16 patches
- **Pre-training**: ImageNet-21k
- **Fine-tuning**: Custom pet breed dataset

### Key Components
1. **Image Preprocessing**: Resize to 224×224, RGB conversion
2. **Patch Embedding**: 16×16 patch extraction
3. **Transformer Encoder**: 12 layers with multi-head attention
4. **Classification Head**: Custom head for 142 breed classes
5. **Data Augmentation**: Rotation, flip, zoom transformations

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AI-Based-Pets-Breed-Recognition.git
cd AI-Based-Pets-Breed-Recognition
```

2. **Install dependencies**
```bash
cd "Cat and Dog Breed Detection With ViT"
pip install -r requirements.txt
```

### Key Dependencies
```
torch==2.5.1+cu121
transformers==4.56.1
tensorflow==2.20.0
opencv-python==4.12.0.88
pillow==11.2.1
scikit-learn==1.6.1
matplotlib==3.10.3
```

## 📁 Project Structure

```
AI-Based-Pets-Breed-Recognition/
├── Cat and Dog Breed Detection With ViT/
│   ├── main.py                           # Main data processing pipeline
│   ├── Training_Vit.py                   # ViT model training
│   ├── predict.py                        # Breed prediction script
│   ├── Evaluating.py                     # Model evaluation
│   ├── data_loader.py                    # Dataset extraction utilities
│   ├── combiningbothdatasets.py          # Dataset merging
│   ├── fixingdatasets.py                 # Dataset preprocessing
│   ├── split.py                          # Train/validation splitting
│   ├── visualize_and_equalize_class_distribution.py  # Class balancing
│   ├── requirements.txt                  # Python dependencies
│   └── saved_model/                      # Trained model storage
│       ├── vit_transformer/              # ViT model files
│       └── trainlabel_mapping.json       # Class label mapping
└── README.md
```

## 🔄 Training Process

### 1. Data Preparation
Run the complete data pipeline:

```bash
python main.py
```

This will:
- Extract Stanford Dogs and Oxford-IIIT datasets
- Fix folder structure and naming conventions
- Merge both datasets into a unified structure
- Split into train/validation sets (80/20)
- Balance class distributions

### 2. Model Training
Train the Vision Transformer:

```bash
python Training_Vit.py
```

**Training Configuration:**
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: Default ViT settings
- **Device**: Auto-detect (CUDA/CPU)
- **Class Balancing**: 200 samples per class

### 3. Model Evaluation
Evaluate the trained model:

```bash
python Evaluating.py
```

**Metrics Provided:**
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)

## 🎯 Prediction

### Single Image Prediction
```bash
python predict.py
```

Make sure to update the image path in `predict.py`:
```python
test_image_path = "your_image.jpg"  # Replace with your image path
```

### Programmatic Usage
```python
from predict import predict_breed

# Predict breed from image
predicted_breed = predict_breed("path/to/image.jpg")
print(f"Predicted breed: {predicted_breed}")
```

## 📈 Performance

### Model Performance
- **Validation Accuracy**: ~87.5%
- **Training Time**: ~2-3 hours (on GPU)
- **Inference Time**: ~100ms per image
- **Model Size**: ~330MB

### Supported Breeds
The model can classify **142 different breeds** including:
- **Dog Breeds**: Golden Retriever, German Shepherd, Labrador, Poodle, etc.
- **Cat Breeds**: Persian, Maine Coon, British Shorthair, Siamese, etc.

### Sample Predictions
```
🐶 Predicted Breed: golden_retriever
🐱 Predicted Breed: persian_cat
```

## 🛠️ Advanced Usage

### Custom Training
Modify training parameters in `Training_Vit.py`:

```python
training_args = TrainingArguments(
    output_dir="./saved_model/vit_output",
    per_device_train_batch_size=16,      # Adjust batch size
    per_device_eval_batch_size=16,
    num_train_epochs=5,                  # Increase epochs
    learning_rate=2e-5,                  # Custom learning rate
    logging_dir="./logs",
    report_to="none"
)
```

### Data Augmentation
Customize data augmentation in the training pipeline:

```python
# Add more augmentation techniques
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `Training_Vit.py`
   - Use gradient accumulation
   - Enable mixed precision training

2. **Dataset Not Found**
   - Ensure `stanford.tar` and `oxford.tar.gz` are in the project directory
   - Check file permissions and extraction paths

3. **Model Loading Errors**
   - Verify `saved_model/` directory exists
   - Check `trainlabel_mapping.json` file integrity

### Performance Optimization

1. **GPU Acceleration**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

2. **Batch Processing**
   ```python
   # Process multiple images at once
   batch_images = [preprocess_image(img) for img in image_paths]
   batch_predictions = model(torch.stack(batch_images))
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stanford Vision Lab** for the Stanford Dogs Dataset
- **Oxford Visual Geometry Group** for the Oxford-IIIT Pet Dataset
- **Hugging Face** for the Transformers library
- **Google Research** for the Vision Transformer architecture

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [Link to detailed docs]

---

⭐ **Star this repository** if you found it helpful!

🐾 **Happy Pet Breed Recognition!**