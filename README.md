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
- **Train/Validation/Test Split**: 70/10/20 ratio
- **Class Balancing**: Equalized to 200 samples per class (where available)

## 🧠 Model Architectures

The project supports multiple state-of-the-art vision models:

### Supported Models
1. **ResNet18** - `microsoft/resnet-18`
2. **ResNet50** - `microsoft/resnet-50`
3. **Vision Transformer (ViT)** - `google/vit-base-patch16-224-in21k`
4. **EfficientNet-B0** - `google/efficientnet-b0`
5. **Swin Transformer** - `microsoft/swin-base-patch4-window7-224`
6. **ConvNeXt** - `facebook/convnext-base-224-22k`

### Key Components
1. **Image Preprocessing**: Resize to 224×224, RGB conversion, EXIF correction
2. **Data Augmentation**: Albumentations (RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter)
3. **Advanced Augmentation**: MixUp and CutMix for regularization
4. **Class Balancing**: Equalized to 200 samples per class
5. **Two-Phase Training**: Freeze/unfreeze strategy for stable fine-tuning
6. **Classification Head**: Custom head for 142 breed classes

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
cd "Cat and Dog Breed Detection"
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
albumentations (for fast augmentation)
accelerate==1.7.0
datasets==3.6.0
```

**Note:** See `requirements.txt` for complete dependency list.

## 📁 Project Structure

```
AI-Based-Pets-Breed-Recognition/
├── Cat and Dog Breed Detection/
│   ├── main.py                           # Main data processing pipeline
│   ├── data_loader.py                    # Dataset extraction utilities
│   ├── fixingdatasets.py                 # Dataset renaming and organization
│   ├── combiningbothdatasets.py          # Dataset merging
│   ├── split.py                          # Train/validation/test splitting
│   ├── model_names.txt                   # Supported model configurations
│   ├── requirements.txt                  # Python dependencies
│   ├── Models/
│   │   ├── Models.py                     # Model runner and configurations
│   │   ├── training_model_withloaddata.py # Main training script
│   │   └── optimized_V_and_E_Class_distribution.py  # Class balancing & augmentation
│   └── saved_model/                      # Trained model storage
│       ├── {ModelName}/                  # Model-specific directories
│       └── trainlabel_mapping.json       # Class label mapping
├── README.md                             # Project documentation
└── report.md                             # Technical preprocessing report
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
- Split into train/validation/test sets (70/10/20)
- Balance class distributions (200 samples per class)

### 2. Model Training
Train any of the supported models:

```bash
cd Models
python Models.py
```

Or train a specific model by modifying `Models.py` to select which models to run.

**Training Configuration:**
- **Epochs**: 12-15 (model-dependent)
- **Batch Size**: 8-16 (model-dependent)
- **Learning Rate**: 1e-5 to 1e-4 (model-dependent)
- **Device**: Auto-detect (CUDA/CPU)
- **Class Balancing**: 200 samples per class
- **Augmentation**: MixUp + CutMix enabled
- **Training Strategy**: Two-phase (freeze/unfreeze)

### 3. Model Evaluation
Model evaluation is performed automatically during training. Detailed metrics are saved after each epoch in the model's save directory:

- **Accuracy**
- **Precision** (weighted average)
- **Recall** (weighted average)
- **F1-Score** (weighted average)
- **Classification Report** (per-class metrics)
- **Confusion Matrix**

Results are saved in:
- `saved_model/{ModelName}/optuna_cv_results.json`
- `saved_model/{ModelName}/final_model/{ModelName}_epoch_{N}_report.txt`

## 📈 Performance

### Model Performance
- **Validation Accuracy**: Varies by model (typically 85-90%+)
- **Training Time**: ~2-4 hours per model (on GPU)
- **Inference Time**: ~50-150ms per image (model-dependent)
- **Model Size**: 50-400MB (model-dependent)

### Supported Breeds
The models can classify **142 different breeds** including:
- **Dog Breeds**: Golden Retriever, German Shepherd, Labrador, Poodle, Chihuahua, Beagle, etc.
- **Cat Breeds**: Persian, Maine Coon, British Shorthair, Siamese, Abyssinian, Bengal, etc.

### Model Comparison
Each model has been optimized with specific hyperparameters. See `Models/Models.py` for detailed configurations.

## 🛠️ Advanced Usage

### Custom Training
Modify training parameters in `Models/Models.py`:

```python
best_configs = {
    "ViT": {
        "model_name": "google/vit-base-patch16-224-in21k",
        "save_dir": "saved_model/ViT",
        "learning_rate": 5e-5,        # Adjust learning rate
        "batch_size": 8,              # Adjust batch size
        "weight_decay": 0.06,
        "mixup_alpha": 0.25,          # MixUp strength
        "cutmix_alpha": 1.0,          # CutMix strength
        "num_train_epochs": 12,       # Adjust epochs
        "warmup_epochs": 4,           # Warmup phase length
    },
}
```

### Data Augmentation
Customize data augmentation in `Models/optimized_V_and_E_Class_distribution.py`:

```python
def get_basic_augmentation(size=(224, 224)):
    return A.Compose([
        A.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        # Add more augmentations here
    ])
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `Models/Models.py` (model configurations)
   - Gradient accumulation is already enabled (steps=2)
   - Mixed precision training (FP16) is enabled by default

2. **Dataset Not Found**
   - Ensure `stanford.tar` and `oxford.tar.gz` are in the project root
   - Run `main.py` to extract and process datasets
   - Check file permissions and extraction paths

3. **Model Loading Errors**
   - Verify `saved_model/` directory exists
   - Check `trainlabel_mapping.json` file integrity
   - Ensure model name matches Hugging Face model identifier

4. **Class Mismatch Errors**
   - Ensure train/val/test splits have matching class structures
   - Re-run `split.py` if classes don't match
   - Check that all breed folders exist in all splits

### Performance Optimization

1. **GPU Acceleration**
   - Automatically detected and used if available
   - Set `CUDA_VISIBLE_DEVICES` to select specific GPU

2. **Batch Size Tuning**
   - Larger models (ViT, ConvNeXt): Use batch_size=8
   - Medium models (Swin, ResNet50): Use batch_size=12
   - Smaller models (ResNet18, EfficientNet): Use batch_size=16

3. **Memory Management**
   - FP16 training reduces memory by ~50%
   - Gradient accumulation allows effective larger batch sizes
   - CUDA cache is cleared between model runs

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

## 📚 Additional Documentation

For detailed information about preprocessing techniques, augmentation strategies, and implementation details, see:
- **[report.md](report.md)** - Comprehensive technical report on all preprocessing steps, techniques, and their advantages

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review the technical report (`report.md`) for implementation details
- Check model configurations in `Models/Models.py`

---

⭐ **Star this repository** if you found it helpful!

🐾 **Happy Pet Breed Recognition!**