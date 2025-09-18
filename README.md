# 🏛️ Vietnamese Landmark Classifier

**AI-Powered Recognition of Vietnam's Iconic Landmarks using Vision Transformer (ViT)**

---

## 🎯 Project Overview

This project implements a state-of-the-art **Vision Transformer (ViT)** model to classify and recognize famous Vietnamese landmarks. The system can accurately identify iconic locations such as Ha Long Bay, Hoi An Ancient Town, Imperial City of Hue, and many more Vietnamese cultural heritage sites.

### 🌟 Key Features

- **🤖 Advanced AI Architecture**: Leverages Vision Transformer (ViT) for superior image classification performance
- **🇻🇳 Vietnamese Heritage Focus**: Specialized dataset covering Vietnam's most significant landmarks
- **🚀 Production-Ready**: Deployed as an interactive web application on Hugging Face Spaces
- **📊 High Accuracy**: Optimized model performance through extensive training and validation
- **🔧 Modular Design**: Clean, maintainable codebase with separated concerns

---

## 🏗️ Architecture & Technology Stack

### Core Technologies
- **Deep Learning Framework**: PyTorch
- **Model Architecture**: Vision Transformer (ViT)
- **Frontend**: Gradio for interactive web interface
- **Deployment**: Hugging Face Spaces
- **Data Processing**: Custom data pipeline with augmentation

### Model Specifications
```python
Model: Vision Transformer (ViT)
Input Resolution: 224x224 RGB images
Architecture: Transformer-based encoder
Training Strategy: Transfer learning + Fine-tuning
Optimization: AdamW optimizer with learning rate scheduling
```

---

## 📁 Project Structure

```
vietnamese-landmark-classifier/
├── 📂 data/                    # Dataset management
│   ├── 📂 test/               # Test dataset
│   ├── 📂 train/              # Training dataset
│   └── 📂 val/                # Sample images for prediction
├── 📂 demo/                   # Demo application
│   ├── 📂 examples/           # Sample images
│   ├── 🐍 app.py             # Gradio web interface
│   ├── 🐍 model.py           # Model loading and inference
│   └── 📄 requirements.txt   # Dependencies
├── 📂 models/                 # Trained model artifacts
├── 📂 modular/               # Core modules
│   ├── 🐍 data_setup.py     # Data loading and preprocessing
│   ├── 🐍 engine.py         # Training engine
│   ├── 🐍 helper.py         # Utility functions
│   ├── 🐍 models.py         # Model architectures
│   └── 🐍 utils.py          # Additional utilities
├── 📂 notebooks/            # Research and experimentation
│   ├── 📓 data_exploration.ipynb
│   ├── 📓 demo_gradio_app.ipynb
│   └── 📓 training_and_explore.ipynb
└── 📂 training/             # Training pipeline
    └── 🐍 train.py          # Main training script
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/huynhtiennhat0403/vietnamese-landmark-classifier.git
cd vietnamese-landmark-classifier
```

### 2. Install Dependencies
```bash
pip install -r demo/requirements.txt
```
### 3. Run train.py to train and save model locally
```bash
python training/train.py
```

### 4. Run the Demo Application
```bash
python demo/app.py
```

### 5. Try Online Demo
Visit our [Hugging Face Space](https://huggingface.co/spaces/HuynhNhat0403/VietnameseLandmarkClassification) for instant testing!