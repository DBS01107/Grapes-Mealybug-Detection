# 🍇 Grape Leaf Mealybug Detection (Unsupervised + Pseudo-Labeling)

This project is an end-to-end pipeline for detecting mealybug infestations on grape leaves using a combination of deep learning feature extraction, unsupervised clustering, and pseudo-labeling techniques. It automates image preprocessing, segmentation, feature extraction, clustering, and classification.

---

## 📌 Features

- ✅ Resizes, denoises, and segments grape leaf images
- ✅ Extracts deep features using a pretrained ResNet50 model
- ✅ Clusters images using KMeans + t-SNE visualization
- ✅ Selects high-confidence samples via pseudo-labeling
- ✅ Trains a lightweight classifier (Logistic Regression)
- ✅ Provides a CLI-based prediction script for new images

---

## 🛠️ Tech Stack

- Python
- PyTorch / torchvision
- scikit-learn
- PIL / OpenCV
- NumPy / pandas
- t-SNE / PCA / KMeans
- Jupyter Notebook
- joblib

---

## 📁 Folder Structure

