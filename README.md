# 🧠 Brain Tumor MRI Image Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95.12%25-brightgreen.svg)

An advanced, interactive web application that classifies brain tumors from MRI scans using Deep Learning. The core of this project relies on **RadImageNet**, a specialized medical imaging model, achieving a stunning **95.12% Test Accuracy**.


## 📖 Project Overview
Detecting brain tumors quickly and accurately is crucial for medical diagnosis. This project automates the classification of brain tumors into four distinct categories:
1. **Glioma** 
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor** (Healthy)

The project includes the complete data science pipeline: Data Exploration, Custom CNN development, Transfer Learning (using MobileNetV2 and RadImageNet), Fine-Tuning, and Deployment via Streamlit.

## 🧬 Model Architecture: The Power of RadImageNet
Initially, a Custom CNN was developed which achieved ~87% accuracy. To push the boundaries of performance, we implemented **Transfer Learning**.

Instead of using a generic ImageNet model (like ResNet50 trained on dogs and cars), this project utilizes weights from **RadImageNet**—a massive dataset of CT, Ultrasound, and MRI scans. By fine-tuning this medically-pretrained ResNet50 architecture on our specific MRI dataset, the model rapidly converged to **95.12% accuracy** without overfitting.

## 📊 Performance Metrics (Test Set)
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Glioma** | 0.99 | 0.97 | 0.98 |
| **Meningioma** | 0.91 | 0.94 | 0.92 |
| **No Tumor** | 0.92 | 0.90 | 0.91 |
| **Pituitary** | 0.98 | 0.98 | 0.98 |

**Overall Accuracy:** 95.12%

## ⚠️ Important Note: MRI Sequence Mismatch
This model was trained exclusively on **T1-Weighted Contrast-Enhanced (T1c)** MRI scans (where the fluid is dark and tumors feature a bright glowing ring). 

**Domain Shift Warning:** If you upload a T2 FLAIR MRI scan (where the entire tumor is a solid, fuzzy white cloud), the model will likely misclassify it as "No Tumor". For accurate predictions, ensure the uploaded images are T1c Axial, Coronal, or Sagittal slices!

## 💻 Installation & Local Setup

### 1. Clone the repository
`Bash
git clone https://github.com/nathdhiman005-svg/Brain-Tumor-MRI-Image-Classification.git
cd Brain-Tumor-MRI-Image-Classification
`

### 2. Pull the Large Model File (Git LFS)
Because the highly accurate 
adimagenet_finetuned.h5 model is over 200MB, it is tracked via Git LFS. Ensure you have Git LFS installed:
`Bash
git lfs install
git lfs pull
`

### 3. Install Dependencies
`bash
pip install -r requirements.txt
`

### 4. Run the Streamlit App
`bash
streamlit run app.py
`

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Pre-trained Weights:** RadImageNet (ResNet50)
* **Data Manipulation:** NumPy, Pandas, OpenCV, Pillow
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
