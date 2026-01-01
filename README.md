# 🌿 Plant Leaf Disease Detection using CNN

An end-to-end **Computer Vision** project that detects plant leaf diseases from images using a **Convolutional Neural Network (CNN)** built with **TensorFlow & Keras**.  
The model classifies leaf images into healthy or diseased categories with high accuracy.

---

## 📌 Project Overview

Plant diseases significantly affect agricultural productivity.  
This project aims to **automate plant disease detection** using image classification, helping farmers and researchers identify diseases early.

---

## 🎯 Objectives

- Build a CNN model for plant leaf disease classification  
- Perform dataset preprocessing and splitting  
- Train and evaluate the model  
- Predict disease from a single leaf image  
- Achieve high test accuracy  

---

## 🧠 Model Used

- **Custom CNN Architecture**
- Convolution + MaxPooling layers
- Fully Connected Dense layers
- Softmax output layer

---

## 📊 Results

| Metric | Value |
|------|------|
| Test Accuracy | **92.02%** |
| Prediction Confidence | Up to **98%+** |
| Model Type | CNN |
| Framework | TensorFlow / Keras |

---

## 🗂️ Project Structure

Plant_Leaf_Disease_Detection/
│
├── data/
│ ├── train/
│ ├── val/
│ └── test/
│
├── model/
│ └── plant_leaf_cnn_model.h5
│
├── train_cnn.py
├── predict.py
├── EDA.ipynb
├── split_dataset.py
├── requirements.txt
└── README.md


---

## 📁 Dataset

- Image dataset containing **healthy and infected leaf images**
- Each class stored in a separate folder
- Dataset split into:
  - Training
  - Validation
  - Testing

> ⚠️ Dataset not uploaded due to large file size.

---

## 🔧 Installation & Setup

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv tf_env
tf_env\Scripts\activate
```
### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
### Training the Model
```bash
python train_cnn.py
```
After training:

Model is saved as
model/plant_leaf_cnn_model.h5

### 🔍 Making Predictions
Run the prediction script:
```bash
python predict.py
```
Enter image path when prompted:
```bash
📸 Enter image path: C:\path\to\leaf_image.jpg
```

### Sample Output:
🌿 Prediction Result
----------------------
🦠 Disease Class : Pepper__bell___healthy
📊 Confidence    : 98.72%

## 🧪 Exploratory Data Analysis (EDA)
Open the notebook:
```bash
EDA.ipynb
```

Includes:
- Class distribution
- Sample images per class
- Image shape analysis

###🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

### 👨‍💻 Author
Sam Johnston C
B.Tech – Artificial Intelligence & Data Science
St. Joseph College of Engineering

###⭐ Acknowledgments

- Kaggle / Public plant disease datasets
- TensorFlow & Keras documentation
- Open-source AI community
