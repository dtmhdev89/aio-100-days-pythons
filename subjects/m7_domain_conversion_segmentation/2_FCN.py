from common_libs import torch, DataLoader, summary
from custom_utils import get_device, \
    display_prediction, evaluate, make_results_dir, \
    model_save_in_safetensors, make_model_dir
from data import PetDataLoader
import torch.nn as nn
import copy
import os


# Down-Sample
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.enc_1 = Encoder(n_channels, 64)
        self.enc_2 = Encoder(64, 128)
        self.enc_3 = Encoder(128, 256)
        self.enc_4 = Encoder(256, 512)
        self.enc_5 = Encoder(512, 512)
        self.enc_6 = Encoder(512, 512)
        self.out_conv = nn.Conv2d(512, 128*128*3, kernel_size=2)

    def forward(self, x):
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        x = self.enc_5(x)
        x = self.enc_6(x)
        x = self.out_conv(x)

        return torch.reshape(x, (-1, 3, 128, 128))


if __name__ == "__main__":
    device = get_device()
    img_size = (128, 128)
    num_classes = 3
    batch_size = 64
    num_workers = 6

    results_path = make_results_dir()

    pet_loader = PetDataLoader(img_size=img_size)
    train_set = pet_loader.get_dataset(split='trainval')
    test_set = pet_loader.get_dataset(split='test')

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers)

    model = UNet(n_channels=3, n_classes=3).to(device)
    summary(model, (3, 128, 128))

    data = torch.randn((1, 3, 128, 128)).to(device)
    output = model(data)
    print(output.shape)

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
    predicted_path = os.path.join(results_path, '2_fcn')

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
        display_prediction(model, display_image, display_target,
                           device, predicted_path)

    train_losses.append(epoch_loss)
    test_losses.append(test_loss)

    model.load_state_dict(best_model_wts)

    model_path = make_model_dir()
    model_save_in_safetensors(model=model,
                              safetensors_path=os.path.join(
                                  model_path,
                                  "2_FCN.safetensors"))
    # torch.save(model.state_dict(), "1.FCN.pt")

    n_test_points = 10

    for i in range(n_test_points):
        img, gt = test_set[i]

        display_prediction(model, img, gt, device, predicted_path)
