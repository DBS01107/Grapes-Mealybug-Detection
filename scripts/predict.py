import os
import argparse
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import joblib

# Load saved model
model = joblib.load(".\models\pseudo_classifier_model.pkl")

# Define image transformation (should match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load ResNet50 as feature extractor
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor).cpu().numpy().flatten()
    return features

def predict(image_path):
    features = extract_features(image_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()
    return prediction, confidence

def main(image_path):
    label, confidence = predict(image_path)
    label_name = "Mealybug" if label == 1 else "Clean"
    print(f"🔍 Prediction: {label_name} ({confidence:.2f} confidence)")
    return label_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("🚫 Invalid image path")
        exit(1)

    main(args.image)
#     label, confidence = predict(args.image)
#     label_name = "Mealybug" if label == 1 else "Clean"
#     print(f"🔍 Prediction: {label_name} ({confidence:.2f}