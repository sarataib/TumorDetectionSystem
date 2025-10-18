# TumorDetectionSystem

## 🧠 Brain Tumor Detection using Deep Learning

A comprehensive deep learning system for detecting brain tumors from MRI images using Convolutional Neural Networks (CNN). This project implements advanced image processing and machine learning techniques to classify brain MRI scans as either containing tumors or being healthy.

## 📊 Project Overview

This system analyzes medical MRI images to distinguish between healthy brain tissue and tumorous regions. Through transfer learning, data augmentation, and optimized CNN architectures, it achieves high accuracy in tumor detection while handling the challenges of limited medical data.

## 🏗️ Project Structure

```
TumorDetectionSystem/
├── .ipynb_checkpoints/              # Jupyter notebook checkpoints
├── augmentation data/               # Augmented dataset
│   ├── yes/                        # Augmented tumor images
│   └── no/                         # Augmented healthy images
├── brain_tumor_dataset/             # Original dataset
│   ├── yes/                        # Original tumor images (155)
│   └── no/                         # Original healthy images (98)
├── logs/                           # Training logs for TensorBoard
├── models/                         # Saved model checkpoints
├── code_source_Detection tumor.ipynb  # Main analysis notebook
├── code_source_application.py      # Streamlit application
├── convnet_architecture.jpg        # Model architecture diagram
├── demonstration_application.mp4   # Demo video
└── requirements.txt                # Project dependencies
```

## 🎯 Key Features

- **High Accuracy**: Achieves 97%+ accuracy on test data
- **Data Augmentation**: Advanced techniques to handle imbalanced datasets
- **Transfer Learning**: Utilizes VGG16 pre-trained models
- **Real-time Prediction**: Web interface for instant tumor detection
- **Medical Image Processing**: Specialized preprocessing for MRI images
- **Model Interpretability**: Comprehensive performance metrics and visualizations

## 📈 Performance Metrics

### Model Performance
- **Test Accuracy**: 97.74%
- **F1 Score**: 0.98
- **Validation Accuracy**: 91.61%
- **Training Time**: Optimized for efficient convergence

### Dataset Statistics
- **Original Dataset**: 253 images (155 tumor, 98 healthy)
- **After Augmentation**: 2,065 images (balanced classes)
- **Train/Val/Test Split**: 70%/15%/15%

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.5

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/sarataib/TumorDetectionSystem.git
cd TumorDetectionSystem
```

2. **Create virtual environment**
```bash
python -m venv tumor_env
source tumor_env/bin/activate  # Linux/Mac
# or
tumor_env\Scripts\activate    # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Jupyter Notebook Analysis
```bash
jupyter notebook code_source_Detection tumor.ipynb
```

### 2. Streamlit Web Application
```bash
streamlit run code_source_application.py
```

### 3. Model Training
The notebook includes complete training pipeline:
- Data loading and preprocessing
- Model architecture definition
- Training with callbacks
- Performance evaluation


### Key Techniques
- **Transfer Learning**: VGG16 base
- **Data Augmentation**: Rotation, shifting, flipping, brightness adjustment
- **Regularization**: Batch normalization, early stopping
- **Optimization**: Adam optimizer with learning rate scheduling

## 📊 Data Processing Pipeline

### 1. Image Preprocessing
```python
def crop_brain_contour(image, plot=False):
    # Converts to grayscale, applies Gaussian blur
    # Thresholding and morphological operations
    # Contour detection and brain extraction
    # Returns cropped brain image
```

### 2. Data Augmentation
- Rotation range: 10°
- Width/height shift: 10%
- Shear range: 10%
- Brightness range: 0.3-1.0
- Horizontal/vertical flip

### 3. Data Splitting
- Training: 1,445 images
- Validation: 310 images  
- Testing: 310 images

## 📈 Results & Evaluation

### Performance Summary
| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | 91.61% | 97.74% |
| F1 Score | 0.987 | 0.979 |
| Loss | 0.275 | 0.082 |

### Confusion Matrix (Best Model)
```
[[ True Neg: 9   False Pos: 1 ]
 [ False Neg: 5  True Pos: 11 ]]
```

## 🎮 Web Application

The project includes a Streamlit web app with:
- **Image Upload**: Drag-and-drop interface
- **Real-time Prediction**: Instant tumor detection
- **Results Display**: Clear classification output
- **User-friendly**: No technical knowledge required

## 🔧 Technical Details

### Dependencies
Core libraries include:
- TensorFlow & Keras for deep learning
- OpenCV for image processing
- scikit-learn for metrics
- Streamlit for web interface
- Matplotlib & Seaborn for visualization



