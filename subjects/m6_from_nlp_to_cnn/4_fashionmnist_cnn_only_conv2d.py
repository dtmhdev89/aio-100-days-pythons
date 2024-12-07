import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary


class DatasetLoader():
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.num_workers = torch.multiprocessing.cpu_count() - 1
        self.fashion_mnist = torchvision.datasets.FashionMNIST
        self.batch_size = 1024
        self.trainloader, self.testloader = self._init_data_loader()

    def _init_data_loader(self):
        trainset = self.fashion_mnist(
            root='data',
            train=True,
            download=True,
            transform=self.transform
        )
        trainloader = DataLoader(trainset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True,
                                 drop_last=True)
        testset = self.fashion_mnist(
            root='data',
            train=False,
            download=True,
            transform=self.transform
        )
        testloader = DataLoader(testset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)

        return trainloader, testloader


class ConvolutionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(ConvolutionModel, self).__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=7)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=7)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=7)
        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=7)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(4*4*hidden_dim*8, hidden_dim * 4)
        self.dense2 = nn.Linear(hidden_dim * 4, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)

        return x


class MainModel():
    def __init__(self, model, device) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def train(self, trainloader, testloader, max_epochs=250):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(max_epochs):
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

                loss.backward()
                self.optimizer.step()

            epoch_accuracy = 100 * running_correct / total
            epoch_loss = running_loss / (i + 1)

            test_loss, test_accuracy = self.evaluate(testloader)
            print(f"Epoch [{epoch + 1}/{max_epochs}], \
                  Loss: {epoch_loss:.4f}, \
                  Accuracy: {epoch_accuracy:.2f}%, \
                  Test Loss: {test_loss:.4f}, \
                  Test Accuracy: {test_accuracy:.2f}%")

            # save for plot
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        return train_losses, train_accuracies, test_losses, test_accuracies

    def evaluate(self, dataloader):
        self.model.eval()
        test_loss = 0.0
        running_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()
        
        accuracy = 100 * running_correct / total
        test_loss = test_loss / len(dataloader)

        return test_loss, accuracy



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_loader = DatasetLoader()
    trainloader, \
        testloader = dataset_loader.trainloader, dataset_loader.testloader
    
    model = ConvolutionModel(input_dim=1,
                             hidden_dim=32,
                             output_dim=10)
    model = model.to(device)

    summary(model, (1, 28, 28))

    main_model = MainModel(model=model, device=device)

    train_losses, train_accuracies, \
        test_losses, test_accuracies = main_model.train(trainloader=trainloader,
                                                        testloader=testloader)
    
    plt.plot(train_losses, label='train_losses')
    plt.plot(test_losses, label='test_losses')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='train_accuracy')
    plt.plot(test_accuracies, label='test_accuracy')
    plt.legend()
    plt.show()
