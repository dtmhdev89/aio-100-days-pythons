import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessing():
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.num_workers = torch.multiprocessing.cpu_count() // 2
        self.trainloader, self.testloader = self._init_loader()

    def _init_loader(self):
        trainset = torchvision.datasets.FashionMNIST(root='data',
                                                     train=True,
                                                     download=True,
                                                     transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=1024,
                                                  num_workers=self.num_workers,
                                                  shuffle=True,
                                                  drop_last=True)
        testset = torchvision.datasets.FashionMNIST(root='data',
                                                    train=False,
                                                    download=True,
                                                    transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=1024,
                                                 num_workers=self.num_workers,
                                                 shuffle=True)

        return trainloader, testloader


class DisplayLoaderImages():
    @classmethod
    def show_img(cls, tensor_img):
        # unnormalize
        tensor_img = tensor_img * 0.5 + 0.5
        np_img = tensor_img.numpy()
        np_img = np.transpose(np_img, (1, 2, 0))
        plt.imshow(np_img)
        plt.show()


class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, ouput_dim, device) -> None:
        super(ClassificationModel, self).__init__()
        self.device = device
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, ouput_dim)
        self.relu = nn.ReLU()
        self._init_weight_and_bias()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def _init_weight_and_bias(self):
        for layer in [self.linear1, self.linear2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

    def fn_evaluate(self, dataloader):
        self.eval()
        test_loss = 0.0
        running_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _confident_score, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

        accuracy = 100 * running_correct / total
        test_loss = test_loss / len(dataloader)

        return test_loss, accuracy

    def do_train(self, trainloader, testloader, max_epochs=250):
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(max_epochs):
            self.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for i, (images, labels) in enumerate(trainloader, 0):
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_correct += (predicted == labels).sum().item()

                loss.backward()
                self.optimizer.step()

            epoch_accuracy = 100 * running_correct / total
            epoch_loss = running_loss / (i + 1)

            test_loss, test_accuracy = self.fn_evaluate(testloader)
            print(f"Epoch [{epoch + 1}/{max_epochs}], \
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loader = DataPreprocessing()
    trainloader, testloader = data_loader.trainloader, data_loader.testloader

    # for images, labels in trainloader:
    #     DisplayLoaderImages.show_img(torchvision.utils.make_grid(images[:8]))
    #     break

    model = ClassificationModel(input_dim=784, hidden_dim=256,
                                ouput_dim=10, device=device)
    train_losses, train_accuracies, \
        test_losses, test_accuracies = model.do_train(trainloader=trainloader,
                                                      testloader=testloader,
                                                      max_epochs=250)

    plt.plot(train_losses, label='train_losses')
    plt.plot(test_losses, label='test_losses')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='train_accuracy')
    plt.plot(test_accuracies, label='test_accuracy')
    plt.legend()
    plt.show()
