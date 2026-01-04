# 🧠 GAG: Gender & Age Group Classification using ResNet50

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A Multi-task Convolutional Neural Network designed to classify gender and age groups from facial images with high accuracy.**

---

## 📸 Demo Preview
<img width="853" height="228" alt="Ảnh màn hình 2025-12-31 lúc 23 21 34" src="https://github.com/user-attachments/assets/f7318863-7bd3-453a-bd2d-b6a5a6d8c4a2" />

## 📖 Introduction
Demographic classification (Gender & Age) is a crucial step in modern **KYC (Know Your Customer)** systems, personalized marketing, and security surveillance. 

This project, **GAG (Gender Age Group)**, implements a **Multi-task Learning** approach using the **ResNet50** architecture. Instead of training two separate models, we use a single backbone to extract features and branch out into two separate output layers:
1.  **Gender Classification:** Binary (Male/Female).
2.  **Age Group Classification:** Multi-class (0-18, 18-30, 30-50, etc.).

## 🚀 Key Features
* **Multi-task Learning:** Efficiently predicts two attributes simultaneously from a single input image.
* **Robust Architecture:** Utilizes **ResNet50** (pre-trained on ImageNet) for powerful feature extraction.
* **Data Augmentation:** Implemented rotation, flipping, and zooming to handle class imbalance and prevent overfitting.
* **High Accuracy:** Optimized using Adam optimizer and dynamic learning rate scheduling.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

## 📂 Dataset
The model is trained on the **UTKFace Dataset**, a large-scale face dataset with long age span (range from 0 to 116 years old).
* **Total Images:** ~20,000+
* **Annotations:** Age, Gender, Ethnicity.
* **Preprocessing:** Images were resized to `128x128` (or `200x200`) and normalized to range `[0, 1]`.

## 🏗️ Model Architecture
The network consists of a shared backbone and two specific heads:

```mermaid
graph TD;
    Input[Input Image] --> ResNet50[ResNet50 Backbone];
    ResNet50 --> Flatten[Flatten/GlobalAveragePooling];
    Flatten --> Dense1[Dense Layer];
    Dense1 --> Branch1[Gender Branch];
    Dense1 --> Branch2[Age Branch];
    Branch1 --> Sigmoid[Sigmoid Activation];
    Branch2 --> Softmax[Softmax Activation];
