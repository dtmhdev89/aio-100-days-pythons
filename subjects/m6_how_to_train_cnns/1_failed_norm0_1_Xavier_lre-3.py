import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 
from torch.optim import Adam


torch.manual_seed(1)


def memoize(func):
    """
    A simple memoization decorator.
    """
    cache = {}

    def wrapper(*args):
        if args in cache:
            print(f'args in cache: {cache}')
            return cache[args]
        result = func(*args)
        cache[args] = result
        print(f'add to cache: {cache}')
        return result
    return wrapper


@memoize
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetLoader():
    def __init__(self) -> None:
        self.transform = self._init_transform()
        self.batch_size = 256
        self.num_workers = torch.multiprocessing.cpu_count() - 1
        self.trainloader, self.testloader = self._init_loader()

    def _init_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform

    def _init_loader(self):
        trainset = CIFAR10(root='data', train=True,
                           download=True, transform=self.transform)
        testset = CIFAR10(root='data', train=False,
                          download=True, transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.batch_size,
                                num_workers=self.num_workers, shuffle=True)

        return trainloader, testloader


class CNNModel(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_layer7 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer9 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_layer10 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()

        self.fc_layer1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Linear(512, n_classes)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.conv_layer8(x)
        x = self.conv_layer9(x)
        x = self.conv_layer10(x)
        x = self.conv_layer11(x)
        x = self.conv_layer12(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        out = self.fc_layer2(x)

        return out


class ModelProcessor():
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def evaluate(self, dataloader):
        self.model.eval()
        test_loss = 0.0
        running_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                # Move inputs and labels to the device
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

        accuracy = 100 * running_correct / total
        test_loss = test_loss / len(dataloader)

        return test_loss, accuracy

    def train(self, trainloader, testloader, max_epoch=50):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(max_epoch):
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * running_correct / total
            epoch_loss = running_loss / (i + 1)

            test_loss, test_accuracy = self.evaluate(testloader)
            print(f"Epoch [{epoch + 1}/{max_epoch}], \
                  Loss: {epoch_loss:.4f}, \
                  Accuracy: {epoch_accuracy:.2f}%, \
                  Test Loss: {test_loss:.4f}, \
                  Test Accuracy: {test_accuracy:.2f}%")

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        return train_losses, train_accuracies, test_losses, test_accuracies


if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    trainloader, \
        testloader = dataset_loader.trainloader, dataset_loader.testloader

    model_processor = ModelProcessor(model=CNNModel(n_classes=10),
                                     device=device())
    train_losses, \
        train_accuracies, \
        test_losses, \
        test_accuracies = model_processor.train(trainloader=trainloader,
                                                testloader=testloader)

    plt.plot(train_losses, label='train_losses')
    plt.plot(test_losses, label='test_losses')
    plt.legend()
