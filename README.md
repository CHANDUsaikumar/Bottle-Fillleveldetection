# 🧴 Beverage Bottle Fill Level Detection using Computer Vision

## 📌 Overview
Maintaining consistent liquid fill levels in beverage bottles is a critical quality control requirement in manufacturing industries. Traditional manual inspection methods are labor-intensive, time-consuming, and prone to human error.

This project implements an **automated computer vision system** using **Convolutional Neural Networks (CNNs)** to detect and classify bottle fill levels from images. The solution aims to improve inspection accuracy, reduce manual effort, and enhance production efficiency.

---

## 🎯 Problem Statement
The objective of this project is to automatically **detect and classify liquid fill levels in beverage bottles** using image-based analysis.

The problem is formulated as a **multi-class image classification task**, where each bottle image is classified into one of the following fill-level categories:
- 30%
- 50%
- 70%
- 80%
- 90%

---

## 🧠 Methodology

### 1. Data Collection
- Captured images of various beverage bottles under controlled lighting conditions
- Multiple bottle types, shapes, and angles were included
- Images were initially captured in HEIC format

### 2. Data Preprocessing
- Converted images from HEIC to JPG for library compatibility
- Image resizing and normalization
- Basic image enhancement using OpenCV

### 3. Data Augmentation
To improve generalization and prevent overfitting, the following augmentation techniques were applied:
- Rotation
- Width and height shifts
- Zooming
- Brightness adjustment
- Fill mode handling

### 4. Model Training
- Implemented a **custom CNN architecture** trained from scratch
- Applied **transfer learning** using VGG16 and ResNet50
- Models were trained and validated on the augmented dataset

### 5. Evaluation
- Performance evaluated using accuracy on a held-out test dataset
- Comparative analysis performed across different architectures

---

## 🏗️ Models Used
- Custom Convolutional Neural Network (CNN)
- VGG16 (Transfer Learning)
- ResNet50 (Transfer Learning)

---

## 📊 Results

| Model | Test Accuracy |
|------|---------------|
| Custom CNN | **90.13%** |
| VGG16 | 85.04% |
| ResNet50 | 66.96% |

**Insight:**  
The custom CNN achieved the highest accuracy due to its suitability for the dataset size and problem complexity. Deeper architectures like ResNet50 showed reduced performance, likely due to over-parameterization.

---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Image Processing:** PIL  
- **Numerical Computing:** NumPy  
- **Environment:** Jupyter Notebook  

---

## 🚧 Challenges
- Variations in image quality due to lighting and camera angles
- HEIC image format incompatibility with standard CV libraries
- Risk of overfitting due to limited dataset size

---

## 🔮 Future Scope
- Extend the system to real-time **video-based fill level detection**
- Improve robustness under varying lighting conditions
- Implement Region of Interest (ROI) extraction
- Optimize the model for **edge deployment** (e.g., NVIDIA Jetson)
- Include additional evaluation metrics such as confusion matrix and precision-recall

---

## ⭐ Conclusion
This project demonstrates a complete end-to-end **computer vision pipeline** for industrial quality inspection. It highlights practical experience in dataset creation, CNN-based modeling, transfer learning, and performance evaluation, making it highly relevant for AI/ML and Computer Vision roles.
