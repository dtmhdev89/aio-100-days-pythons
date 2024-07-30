import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

MODEL_NAME = "/Users/jrvn-hieu/Downloads/model_weights.pth" # replace this with your weight file path

CLASS_NAMES = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]

def predict(model_name, input_image):
    model = models.resnet152(weights=True)
    model.fc = nn.Linear(in_features=2048, out_features=len(CLASS_NAMES), bias=True)
    weights = torch.load(model_name, map_location='cpu')
    model.load_state_dict(weights)

    prep_img_mean = [0.485, 0.456, 0.406]
    prep_img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=prep_img_mean, std=prep_img_std)
    ])

    image = Image.open(input_image)
    if image.format == 'PNG':
        np_image = np.array(image)
        cv2.imwrite('processing_image.jpg', np_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        image = Image.open('processing_image.jpg')

    preproceeded_image = transform(image).unsqueeze(0)

    model.eval()
    output = model(preproceeded_image)
    pred_idx = torch.argmax(output, dim=1)
    predicted_class = CLASS_NAMES[pred_idx]

    return predicted_class

def main():
    st.title("Cervical Cancer Classification App")
    image_file = st.file_uploader("Choose an image to classify", type=['jpg', 'jpeg', 'png'])
    btn_predict = st.button("Predict")

    col1, col2 = st.columns(2)
    
    if btn_predict & (image_file is not None):
        with col1:
            predict_class = predict(MODEL_NAME, image_file)
            st.write(f"**{predict_class}**")

        with col2:
            st.image(image_file)

if __name__ == "__main__":
    main()

