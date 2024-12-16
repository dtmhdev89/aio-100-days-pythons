import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WeatherDataset(Dataset):
    def __init__(self, X, y, transform=None) -> None:
        self.transform = transform
        self.img_paths = X
        self.labels = y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]   


def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0

    return normalized_img


def load_weather_dataset(X_train, X_val,
                         X_test, y_train,
                         y_val, y_test,
                         transform):
    base_path = os.path.join('proccedded_dataset')
    os.makedirs(base_path, exist_ok=True)
    predefined_path = {
        'train': os.path.join(base_path, 'train_dataset.pth'),
        'val': os.path.join(base_path, 'val_dataset.pth'),
        'test': os.path.join(base_path, 'test_dataset.pth'),
    }
    dataset = {
        'train': None,
        'val': None,
        'test': None
    }
    inputs = {
        'train': [X_train, y_train],
        'val': [X_val, y_val],
        'test': [X_test, y_test]
    }

    for dataset_type, dataset_path in predefined_path.items():
        if os.path.exists(dataset_path):
            dataset[dataset_type] = torch.load(predefined_path[dataset_type])
        else:
            dataset[dataset_type] = WeatherDataset(
                *inputs[dataset_type],
                transform=transform
            )
            torch.save(dataset[dataset_type], predefined_path[dataset_type])

    return dataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += self.downsample(shortcut)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, residual_block, n_blocks_lst, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.create_layer(residual_block, 64, 64, n_blocks_lst[0], 1)
        self.conv3 = self.create_layer(residual_block, 64, 128, n_blocks_lst[1], 2)
        self.conv4 = self.create_layer(residual_block, 128, 256, n_blocks_lst[2], 2)
        self.conv5 = self.create_layer(residual_block, 256, 512, n_blocks_lst[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, n_classes)

    def create_layer(self, residual_block, in_channels, out_channels,
                     n_blocks, stride):
        blocks = []
        first_block = residual_block(in_channels, out_channels, stride)
        blocks.append(first_block)

        for _idx in range(1, n_blocks):
            block = residual_block(out_channels, out_channels, stride)
            blocks.append(block)

        block_sequential = nn.Sequential(*blocks)

        return block_sequential

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []

        model.train()
        for _idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model, val_loader,
            criterion, device
        )
        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}')

    return train_losses, val_losses

if __name__ == "__main__":
    seed = 59
    set_seed(seed)
    root_dir = os.path.join('weather-dataset/dataset')
    classes = {
        label_idx: class_name
        for label_idx, class_name in enumerate(
            sorted(os.listdir(root_dir))
        )
    }

    img_paths = []
    labels = []
    for label_idx, class_name in classes.items():
        class_dir = os.path.join(root_dir, class_name)
        for img_filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_filename)
            img_paths.append(img_path)
            labels.append(label_idx)

    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        img_paths, labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    dataset = load_weather_dataset(X_train=X_train, y_train=y_train,
                                   X_val=X_val, y_val=y_val,
                                   X_test=X_test, y_test=y_test,
                                   transform=transform)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_batch_size = 512
    test_batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    train_features, train_labels = next(iter(train_loader))
    print(f'Feature batch shape: {train_features.size()}')
    print(f'Labels batch shape: {train_labels.size()}')
    # img = train_features[0].permute(1, 2, 0)
    # label = train_labels[0].item()
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f'Label: {classes[label]}')
    # plt.show()

    n_classes = len(list(classes.keys()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResNet(
        residual_block=ResidualBlock,
        n_blocks_lst=[2, 2, 2, 2],
        n_classes=n_classes
    ).to(device)

    model.eval()

    dummy_tensor = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_tensor)

    print('Output shape:', output.shape)

    lr = 1e-2
    epochs = 25

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(val_losses, color='orange')
    ax[1].set_title('Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.show()

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    loss_path = os.path.join('results')
    os.makedirs(loss_path, exist_ok=True)
    loss_file_path = os.path.join(loss_path, 'losses.pkl')
    with open(loss_file_path, 'ab') as f:
        pickle.dump('Train Losses and Val Losses:', f)
        pickle.dump([train_losses, val_losses], f)
        pickle.dump('Val Losses and Val Acc:', f)
        pickle.dump([val_loss, val_acc], f)
        pickle.dump('Test Losses and Test Acc:', f)
        pickle.dump([test_loss, test_acc], f)

    print('Evaluation on val/test dataset')
    print('Val accuracy: ', val_acc)
    print('Test accuracy: ', test_acc)
