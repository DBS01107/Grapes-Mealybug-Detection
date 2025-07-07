# Grapes Mealybug Detection

Automated detection of mealybug infestations on grape leaves using deep feature extraction and pseudo-labeling.

---

## Project Overview

- Developed an automated pipeline to detect mealybug infestations in grape leaves using deep feature extraction and unsupervised clustering.
- Preprocessed images with resizing, denoising, and HSV-based segmentation to isolate leaf regions.
- Extracted features via ResNet50, clustered images using KMeans + t-SNE, and applied pseudo-labeling to train a lightweight classifier.
- Achieved 87% validation accuracy.
- Deployed a Streamlit app for real-time, user-friendly predictions with confidence scores.

---

## File Structure

    ├── data/ # Processed datasets and segmented images
    ├── models/ # Trained models and weights
    ├── scripts/ # Prediction and Streamlit app scripts
    ├── README.md # Project overview and instructions
    ├── requirements.txt # Python dependencies

---

## 📌 Features

- ✅ Resizes, denoises, and segments grape leaf images for accurate analysis  
- ✅ Extracts deep features using a pretrained ResNet50 model  
- ✅ Applies KMeans clustering with t-SNE for visualization and data exploration  
- ✅ Utilizes pseudo-labeling to select high-confidence samples for training  
- ✅ Trains a lightweight classifier (Logistic Regression) on extracted features  
- ✅ Provides an interactive Streamlit app for real-time image upload and prediction with confidence scores  
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
- Streamlit

---
## Usage
- Run predictions from script:
    ```bash
        python scripts/predict.py --image path/to/image.jpg
    ---
- Launch Streamlit app:
    ```bash
        streamlit run scripts/streamlit_app.py

---

## Model Details
Feature extractor: Pretrained ResNet50 (last layer removed).

Classifier: Pseudo-labeled classifier trained on deep features.

Input image size: 224x224 pixels.

Outputs: Prediction label ("Mealybug" or "Clean") with confidence score.

