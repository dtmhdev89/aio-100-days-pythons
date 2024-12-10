
import os
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from mnist_number_dataset import LeNetClassifier


@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True,
                                           map_location=torch.device('cpu')))
    lenet_model.eval()

    return lenet_model


def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size

    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)

    with torch.no_grad():
        predictions = model(img_new)

    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)

    return p_max.item()*100, yhat.item()


def main():
    model_path = os.path.join('model')
    test_images_path = os.path.join('images')
    model_state_path = os.path.join(model_path, 'lenet_model.pt')
    model = load_model(model_state_path)

    st.title('Digit Recognition')
    st.subheader('Model: LeNet. Dataset: MNIST')
    option = st.selectbox('How would you like to give the input?',
                          ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image of a digit",
                                type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the digit \
                       {label} with {p:.2f} % probability.")

    elif option == "Run Example Image":
        image = Image.open(os.path.join(test_images_path, 'demo_8.png'))
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The image is of the digit {label} with {p:.2f} \
                   % probability.")

if __name__ == '__main__':
    main()
