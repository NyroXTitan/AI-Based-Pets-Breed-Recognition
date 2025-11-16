import joblib
import numpy as np
import cv
import os
import torch
import json
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

IMG_SIZE = (224, 224)

def preprocess_image(img_path, processor):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

def predict_breed(img_path):
    # Load label map
    with open("saved_model/trainlabel_mapping.json") as f:
        label_map = json.load(f)
    idx_to_class = {v: k for k, v in label_map.items()}

    # Load model and processor
    model_path = "saved_model/vit_transformer"
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess image
    pixel_values = preprocess_image(img_path, processor).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        predicted_idx = torch.argmax(logits, dim=1).item()
        predicted_label = idx_to_class[predicted_idx]

    print(f"🐶 Predicted Breed: {predicted_label}")

if __name__ == "__main__":
    test_image_path = "download.jpg"  # Replace with your test image path
    predict_breed(test_image_path)

