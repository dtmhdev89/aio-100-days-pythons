from common_libs import torch, np, plt, DataLoader, summary
from data import PetDataLoader, create_target_transform
from custom_utils import get_device, get_timestamp
import torch.nn as nn
import copy
import os


def de_normalize(img,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)

    return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = ConvBlock(n_channels, 64, kernel_size=9)
        self.conv2 = ConvBlock(64, 64, kernel_size=9)
        self.conv3 = ConvBlock(64, 64, kernel_size=5)
        self.conv4 = ConvBlock(64, 64, kernel_size=5)
        self.conv5 = ConvBlock(64, 64, kernel_size=5)
        self.conv6 = nn.Conv2d(64, n_classes, kernel_size=3, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x


@torch.inference_mode()
def display_prediction(model, image, target, device, display_prediction):
    model.eval()
    img = image[None, ...].to(device)
    output = model(img)
    pred = torch.argmax(output, axis=1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("Input Image")
    plt.imshow(de_normalize(image.numpy().transpose(1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("Prediction")
    plt.imshow(pred.cpu().squeeze())

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Ground Truth")
    plt.imshow(target)

    plt.savefig(
        os.path.join(display_prediction, f'predicted_img_${get_timestamp()}')
    )
    # plt.show()


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    return test_loss


if __name__ == "__main__":
    results_path = os.path.join('results')
    os.makedirs(results_path, exist_ok=True)

    img_size = (128, 128)
    num_classes = 3
    num_workers = 6
    device = get_device()

    pet_data = PetDataLoader(img_size=img_size)
    # target_transform = create_target_transform(img_size)
    train_set = pet_data.get_dataset(split="trainval")
    test_set = pet_data.get_dataset(split="test")

    batch_size = 64
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers
    )

    index = 2
    test_image = train_set[index][0].numpy().transpose(1, 2, 0)
    test_groundtruth = train_set[index][1]

    de_test_image = de_normalize(test_image)
    print(de_test_image.min())
    print(de_test_image.max())

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Input Image")
    plt.imshow(de_test_image)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title("Target")
    plt.imshow(test_groundtruth)

    # plt.show()

    model = UNet(n_channels=3, n_classes=3).to(device)
    summary(model, (3, 128, 128))

    max_epoch = 30
    LR = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    test_index = 80
    display_image = test_set[test_index][0]
    display_target = test_set[test_index][1]

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    prediction_results = os.path.join(results_path, "1_cnn_segment")
    os.makedirs(prediction_results, exist_ok=True)

    model.to(device)
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(train_loader)
        test_loss = evaluate(model, test_loader, criterion, device)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch [{epoch + 1}/{max_epoch}], Trainning loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Test image_{test_index} after epoch {epoch+1}: ")
        display_prediction(model, display_image, display_target, device, prediction_results)

    train_losses.append(epoch_loss)
    test_losses.append(test_loss)
