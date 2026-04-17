# 🐱🐶 Cat vs Dog Image Classification (MobileNetV2)

A deep learning project that classifies images as Cat or Dog using Transfer Learning with MobileNetV2 in TensorFlow/Keras.

## 🚀 Project Overview

This project uses a pre-trained MobileNetV2 model (trained on ImageNet) and fine-tunes it for binary image classification.

📦 Dataset: Cats vs Dogs (train/test folders)
🧠 Model: MobileNetV2 (Transfer Learning)
🖼️ Input Size: 128 × 128 × 3
⚡ Training optimized for CPU (lightweight setup)
📁 Project Structure
├── train/                 # Training images (cats & dogs)
├── test/                  # Testing images
├── cat.jfif               # Sample cat image
├── dog.jfif               # Sample dog image
├── model_training.py      # Training script
├── README.md              # Project documentation
└── cat_dog_cnn_model.keras  # Saved trained model

## ⚙️ Installation

Install required dependencies:

pip install tensorflow matplotlib numpy opencv-python

## 🧹 Data Preprocessing

Images loaded using image_dataset_from_directory
Resized to 128×128
Normalized (scaled between 0–1)
Optimized pipeline:
cache()
shuffle()
prefetch()

## 🧠 Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet)
Base model is frozen (no retraining)
Custom classifier layers added:
MobileNetV2 (Frozen)
↓
GlobalAveragePooling2D
↓
Dense (128, ReLU)
↓
BatchNormalization
↓
Dropout (0.2)
↓
Dense (64, ReLU)
↓
Dropout (0.2)
↓
Dense (1, Sigmoid)

## 🏋️ Training Details

Loss Function: binary_crossentropy
Optimizer: Adam
Epochs: 10
Steps per epoch: 100 (fast training)
Validation steps: 20

## 📊 Results

✅ Training Accuracy: ~95–96%
✅ Validation Accuracy: up to ~97%
📉 Loss decreases consistently

## 📈 Visualization

Training performance is visualized using:

Accuracy vs Epochs
Loss vs Epochs
🔍 Prediction Pipeline
⚠️ Important Fix (Key Learning)

Initially predictions were wrong due to incorrect preprocessing.

## ✅ Correct Approach

Use MobileNetV2 preprocessing:

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

Steps:

Read image using OpenCV
Convert BGR → RGB
Resize to 128×128
Expand dimensions
Apply preprocess_input
## 🧪 Sample Predictions
Image	Prediction
Cat	~0.0001 (Cat)
Dog	~0.9999 (Dog)

## 💾 Model Saving
model.save('cat_dog_cnn_model.keras')
## ⚡ Key Highlights
🚀 Transfer Learning = Faster training + better accuracy
🧠 MobileNetV2 = Lightweight & efficient
🛠️ Proper preprocessing is critical
💡 Works well even on CPU
