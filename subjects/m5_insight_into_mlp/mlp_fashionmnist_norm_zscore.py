'''
Activation: ReLU
Optimizer: SGD
Initialization: he init
Normalization: zscore (mean, std)
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class MLP():
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        self.input_dim, self.hidden_dim, self.output_dim\
            = input_dim, hidden_dim, output_dim
        self.device = self._get_device()
        self.num_workers = torch.multiprocessing.cpu_count() // 2
        self.trainset, \
            self.trainloader, \
            self.testset, \
            self.testloader = self._load_dataset_and_normalize()
        self.model = self._instantialize_model()
        self.criterion, self.optimizer = self._init_criterion_and_optimizer()

    def _get_device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _load_dataset_and_normalize(self):
        mean, std = self._get_mean_and_std()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))])
        trainset = torchvision.datasets.FashionMNIST(root='data',
                                                     train=True,
                                                     download=True,
                                                     transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=256,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )
        testset = torchvision.datasets.FashionMNIST(root='data',
                                                    train=False,
                                                    download=True,
                                                    transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=256,
            num_workers=self.num_workers,
            shuffle=True
        )

        return trainset, trainloader, testset, testloader

    def _get_mean_and_std(self):
        compute_transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.FashionMNIST(
            root='data',
            train=True, 
            transform=compute_transform,
            download=True)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=256,
                                             shuffle=False,
                                             num_workers=self.num_workers)
        mean = 0.0
        for images, _ in loader:
            batch_samples = images.size(0)  # Batch size
            # like reshape
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(loader.dataset)
        variance = 0.0
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            variance += ((images - mean.unsqueeze(1))**2).sum([0, 2])

        std = torch.sqrt(variance / (len(loader.dataset)*28*28))

        return mean, std

    def show_dataset(self, to_pos=1, train=True):
        iteration = self.trainloader if train else self.testloader

        for i, (images, _labels) in enumerate(iteration, 0):
            self._show_img(torchvision.utils.make_grid(images[:8]))
            if i == (to_pos - 1):
                break

    def _show_img(self, img):
        img = img * 0.5 + 0.5
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    def _instantialize_model(self):
        model = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        model = model.to(self.device)

        return model

    def _init_criterion_and_optimizer(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2)

        return criterion, optimizer

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_loss = test_loss / len(self.testloader)

        return test_loss, accuracy

    def train(self, epochs=50):
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        for epoch in range(epochs):
            running_loss, running_correct, total = 0.0, 0, 0

            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                # Determine class prediction and track accuracy
                _confident_score, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

                loss.backward()
                self.optimizer.step()

            epoch_accuracy = 100 * running_correct / total
            epoch_loss = running_loss / (i + 1)
            test_loss, test_accuracy = self.evaluate()
            print(f"Epoch \
                [{epoch + 1}/{epochs}], \
                Loss: {epoch_loss:.4f}, \
                Accuracy: {epoch_accuracy:.2f}%, \
                Test Loss: {test_loss:.4f}, \
                Test Accuracy: {test_accuracy:.2f}%")

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        return train_losses, train_accuracies, test_losses, test_accuracies

    def plot_result(self, train_losses, train_accuracies,
                    test_losses, test_accuracies):
        plt.plot(train_losses, label='train_losses')
        plt.plot(test_losses, label='test_losses')
        plt.legend()
        plt.show()

        plt.plot(train_accuracies, label='train_accuracy')
        plt.plot(test_accuracies, label='test_accuracy')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    input_dim, hidden_dim, output_dim = 784, 256, 10
    mlp = MLP(input_dim, hidden_dim, output_dim)
    # mlp.show_dataset(train=True, to_pos=1)

    test_loss, test_accuracy = mlp.evaluate()
    print(f'test_loss: {test_loss}')
    print(f'test_accuracy: {test_accuracy}')

    train_losses, train_accuracies, \
        test_losses, test_accuracies = mlp.train(epochs=100)
    mlp.plot_result(train_losses, train_accuracies,
                    test_losses, test_accuracies)
