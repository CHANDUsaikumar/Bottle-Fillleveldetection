# Bottle-Fillleveldetection
🧴 Beverage Bottle Fill Level Detection using Computer Vision
📌 Overview

Ensuring consistent liquid fill levels in beverage bottles is a critical quality control task in the manufacturing industry. Manual inspection is time-consuming, error-prone, and not scalable.

This project presents an automated computer vision–based system using Convolutional Neural Networks (CNNs) to detect and classify bottle fill levels from images. The system is designed to improve inspection accuracy, reduce human dependency, and enable scalable quality assurance in production environments.

🎯 Problem Statement

To automatically detect and classify the fill level of liquid in beverage bottles under controlled conditions using image-based analysis.

The task is formulated as a multi-class image classification problem, where each image is classified into one of the predefined fill levels:

30%

50%

70%

80%

90%

🧠 Solution Approach

The system follows a complete end-to-end computer vision pipeline:

Data Collection

Captured images of different bottle types

Controlled lighting and background setup

Multiple viewing angles and fill levels

Preprocessing

Converted images from HEIC to JPG

Image resizing and normalization

Quality enhancement using OpenCV

Data Augmentation

Rotation

Width & height shifts

Zooming

Brightness adjustment

Fill mode handling
This helps improve model generalization and reduce overfitting.

Model Training

Custom CNN architecture

Transfer learning with VGG16

Transfer learning with ResNet50

Evaluation

Accuracy-based evaluation on a held-out test set

Comparative analysis across architectures

🏗️ Model Architectures Used

Custom CNN (trained from scratch)

VGG16 (transfer learning)

ResNet50 (transfer learning)

📊 Results
Model	Test Accuracy
Custom CNN	90.13%
VGG16	85.04%
ResNet50	66.96%

Observation:
The custom CNN performed best due to better alignment with dataset size and task complexity. Deeper architectures like ResNet50 showed signs of over-parameterization for the available dataset.

⚙️ Technologies & Tools

Programming Language: Python

Deep Learning: TensorFlow, Keras

Computer Vision: OpenCV

Image Processing: PIL

Numerical Computing: NumPy

Development Environment: Jupyter Notebook

🚧 Challenges Faced

Image quality variations due to lighting and angle differences

HEIC image format incompatibility, requiring conversion to JPG

Overfitting risk due to limited dataset size

🔮 Future Improvements

Extend the system to real-time video-based fill level detection

Implement Region of Interest (ROI) extraction for better robustness

Improve performance under varying lighting conditions

Optimize the model for edge deployment (e.g., NVIDIA Jetson)

Add detailed metrics such as confusion matrix and precision-recall analysis

📁 Project Structure
├── dataset/
│   ├── train/
│   ├── test/
│
├── notebooks/
│   └── bottle_fill_detection.ipynb
│
├── models/
│   ├── cnn_model.h5
│   ├── vgg16_model.h5
│   └── resnet50_model.h5
│
├── README.md

⭐ Why This Project Matters

This project demonstrates:

Practical application of Computer Vision in industrial automation

End-to-end ML pipeline development

Hands-on experience with CNNs and transfer learning

Strong alignment with AI/ML and Computer Vision roles
