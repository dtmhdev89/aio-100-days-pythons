import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def show_img(img):
    print(img.shape)
    img = img / 255.0
    np_img = img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    plt.imshow(np_img)
    plt.show()


def evaluate(model, criterion, testloader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 % correct / total
    test_loss = test_loss / len(testloader)

    return test_loss, accuracy


def train(model, optimizer, criterion,
          trainloader, testloader, max_epochs, device):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(max_epochs):
        running_loss = 0
        running_correct = 0
        total = 0
        count = 0

        for i, (inputs, labels) in enumerate(trainloader, 0):
            count += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _confident_score, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()

        epoch_accuracy = 100 * running_correct / total
        print(f'i value outside of loop loader: {i} - {count}')
        epoch_loss = running_loss / (i + 1)
        test_loss, test_accuracy = evaluate(model=model,
                                            criterion=criterion,
                                            testloader=testloader,
                                            device=device)
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = torch.multiprocessing.cpu_count() // 2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1.0/255.0,))])
    trainset = torchvision.datasets.FashionMNIST(root='data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=1024,
                                              num_workers=num_workers,
                                              shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root='data',
                                                download=True,
                                                train=False,
                                                transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1024,
                                             num_workers=num_workers,
                                             shuffle=True)

    for i, (images, labels) in enumerate(trainloader, 0):
        show_img(torchvision.utils.make_grid(images[:8]))
        break

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 10)
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    test_loss, test_accuracy = evaluate(model=model,
                                        criterion=criterion,
                                        testloader=testloader,
                                        device=device)
    print(f'test_loss: {test_loss}')
    print(f'test_accuracy: {test_accuracy}')

    max_epoch = 100
    train_losses, train_accuracies, test_losses, test_accuracies = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        trainloader=trainloader,
        testloader=testloader,
        max_epochs=max_epoch,
        device=device
    )

    plt.plot(train_losses, label='train_losses')
    plt.plot(test_losses, label='test_losses')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='train_accuracy')
    plt.plot(test_accuracies, label='test_accuracy')
    plt.legend()
    plt.show()
