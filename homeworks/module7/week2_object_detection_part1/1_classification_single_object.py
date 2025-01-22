from custom_utils import get_optional_args, \
    model_save_in_safetensors, load_model, de_normalize, \
    get_timestamp, REVERSE_LABELS
    
import kagglehub
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import ResNet18_Weights
import copy


# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, annotations_dir, image_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self.filter_images_with_multiple_objects()

    def filter_images_with_multiple_objects(self):
        valid_image_files = []
        for f in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, f)):
                img_name = f
                annotation_name = os.path.splitext(img_name)[0] + ".xml"
                annotation_path = os.path.join(self.annotations_dir, 
                                               annotation_name)

                # Keep images that have single object
                if self.count_objects_in_annotation(annotation_path) <= 1:
                    valid_image_files.append(img_name)
                else:
                    print(
                        f"Image {img_name} has multiple objects and will be excluded from the dataset"
                    )
        return valid_image_files

    def count_objects_in_annotation(self, annotation_path):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            count = 0
            for _obj in root.findall("object"):
                count += 1
            return count
        except FileNotFoundError:
            return 0

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Annotation path
        annotation_name = os.path.splitext(img_name)[0] + ".xml"
        annotation_path = os.path.join(self.annotations_dir, annotation_name)

        # Parse annotation
        label = self.parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        label = None
        for obj in root.findall("object"):
            name = obj.find("name").text
            if (
                label is None
            ):  # Take the first label for now. We are working with 1 label per image
                label = name

        # Convert label to numerical representation (0 for cat, 1 for dog)
        label_num = 0 if label == "cat" else 1 if label == "dog" else -1

        return label_num


if __name__ == "__main__":
    sys_options = get_optional_args()
    model_weight_path = os.path.join('model')
    os.makedirs(model_weight_path, exist_ok=True)
    save_model_path = os.path.join(model_weight_path,
                                   "1_cls_single_obj.safetensors")

    dataset_name = "andrewmvd/dog-and-cat-detection"
    data_dir = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", data_dir)

    # Data directory
    annotations_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')

    # Get list of image files and create a dummy dataframe to split the data
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]
    df = pd.DataFrame({'image_name': image_files})
    print(df.head(2))

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = ImageDataset(
        annotations_dir,
        image_dir,
        transform=transform
    )
    val_dataset = ImageDataset(
        annotations_dir,
        image_dir,
        transform=transform
    )

    # Filter datasets based on train_df and val_df
    train_dataset.image_files = [
        f for f in train_dataset.image_files
        if f in train_df['image_name'].values
    ]
    val_dataset.image_files = [
        f for f in val_dataset.image_files
        if f in val_df['image_name'].values
    ]

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if sys_options['train']:
        # Model
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: cat and dog

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Show model summary
        print(model)

        # Training Loop
        num_epochs = 10
        best_model_weights = copy.deepcopy(model.state_dict())
        best_eval_acc = float(0)

        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                scores = model(data)
                loss = criterion(scores, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                eval_acc = float(0)
                for data, targets in val_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    scores = model(data)
                    _, predictions = scores.max(1)
                    correct += (predictions == targets).sum()
                    total += targets.size(0)

                eval_acc = float(correct) / float(total)
                if best_eval_acc < eval_acc:
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_eval_acc = eval_acc

            print(f'Epoch {epoch+1}/{num_epochs}, \
                  Validation Accuracy: {float(correct)/float(total)*100:.2f}%')

        model.load_state_dict(best_model_weights)
        model_save_in_safetensors(model, save_model_path)

    if sys_options['inference']:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: cat and dog

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        load_model(model, save_model_path, strict=True, device=device)
        model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                scores = model(batch_images)
                predictions = scores.max(1)
                plt.figure(figsize=(10, 8))
                max_img_in_a_row = 4
                rows = int((len(batch_images) // max_img_in_a_row) + 1)
                col = 1
                for img, cls in zip(batch_images, predictions.indices):
                    plt.subplot(rows, max_img_in_a_row, col)
                    col += 1
                    plt.axis('off')
                    img_height, img_width, _ = img.shape
                    x_margin = img.shape[1] * 0.05  # 5% margin from the left
                    y_margin = img.shape[0] * 0.05  # 5% margin from the top
                    # Get text bounding box (to calculate text height)
                    # and add padding for text background
                    text_str = f'Prediction: {REVERSE_LABELS[int(cls)]}'
                    bbox = dict(facecolor='while', edgecolor='none', pad=2)
                    text_bbox = plt.gca().text(x_margin, y_margin, text_str,
                                               color='white', fontsize=12,
                                               bbox=bbox).get_window_extent()
                    text_height = text_bbox.height / plt.gcf().dpi

                    # Calculate y-coordinate for bottom margin
                    # calculate y with dpi
                    y = img_height - y_margin - text_height * plt.gcf().dpi
                    plt.text(x=x_margin, y=y,
                             s=text_str,
                             color='black',
                             bbox=bbox)
                    plt.imshow(de_normalize(img.numpy().transpose(1, 2, 0)))

                result_path = os.path.join('results')
                os.makedirs(result_path, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        result_path,
                        f'predicted_img_{get_timestamp()}'
                    )
                )
                break
