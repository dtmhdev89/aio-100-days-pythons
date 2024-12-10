import os
import time
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ROOT = './data'
BATCH_SIZE = 256
N_IMAGES = 25
VALID_RATIO = 0.9


def hash_function_args(*args, **kwargs):
    """
    Creates a hash of the given function arguments.

    Args:
    *args: Positional arguments.
    *kwargs: Keyword arguments.

    Returns:
    str: The hash of the arguments.
    """
    # Convert arguments to hashable types
    # Convert positional arguments to a tuple of strings
    hashable_args = tuple(map(str, args))
    # Convert keyword arguments to a tuple of sorted key-value strings
    hashable_kwargs = tuple(sorted(f"{k}:{v}" for k, v in kwargs.items()))

    # Combine arguments into a single hashable object
    arg_tuple = hashable_args + hashable_kwargs

    # Create a hash of the argument tuple
    hash_object = hashlib.sha256()
    hash_object.update(str(arg_tuple).encode('utf-8'))
    hash_value = hash_object.hexdigest()

    return hash_value


def memoize(func):
    """
    Simple Memoization Decorator
    """
    cache = {}

    def wrapper(*args, **kwargs):
        hashkey = hash_function_args(*args, kwargs=kwargs)

        if hashkey in cache:
            return cache[hashkey]

        result = func(*args, **kwargs)
        cache[hashkey] = result
        print(f'add to cache: {cache}')
        return result

    return wrapper

@memoize
def device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def plot_images(images):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i], cmap='bone')
        ax.axis('off')


def compute_mean_and_std_on_dataset(dataset):
    mean = dataset.data.float().mean() / 255
    std = dataset.data.float().std() / 255

    return mean, std


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6,
            kernel_size=5, padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)
        self._prepare_blocks()

    def _prepare_blocks(self):
        outer_self = self

        self.block_1 = nn.Sequential(
            outer_self.conv1,
            outer_self.avgpool1,
            nn.ReLU()
        )
        self.block_2 = nn.Sequential(
            outer_self.conv2,
            outer_self.avgpool2,
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            outer_self.flatten,
            outer_self.fc_1,
            outer_self.fc_2,
            outer_self.fc_3
        )

    def forward(self, inputs):
        outputs = self.block_1(inputs)
        outputs = self.block_2(outputs)
        outputs = self.dense(outputs)

        return outputs


def train(model, optimizer, criterion,
          train_dataloader, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device())
        labels = labels.to(device())

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches \
                 | accuracy {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_acc / total_count
                ),
                f"| Elapsed Time: {elapsed}"
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss


def evaluate(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device())
            labels = labels.to(device())

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss


def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows = 1, ncols =2 , figsize = (12,6))
    axs[0].plot(epochs, train_accs, label = "Training")
    axs[0].plot(epochs, eval_accs, label = "Evaluation")
    axs[1].plot(epochs, train_losses, label = "Training")
    axs[1].plot(epochs, eval_losses, label = "Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()


def plot_filtered_images(images, filters):
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    filters = filters.cpu()

    n_images = images.shape[0]
    n_filters = filters.shape[0]

    filtered_images = F.conv2d(images, filters)

    fig = plt.figure(figsize=(20, 10))

    for i in range(n_images):

        ax = fig.add_subplot(n_images, n_filters+1, i+1+(i*n_filters))
        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title('Original')
        ax.axis('off')

        for j in range(n_filters):
            image = filtered_images[i][j]
            ax = fig.add_subplot(n_images, n_filters+1, i+1+(i*n_filters)+j+1)
            ax.imshow(image.numpy(), cmap='bone')
            ax.set_title(f'Filter {j+1}')
            ax.axis('off')


if __name__ == "__main__":
    train_data = datasets.MNIST(
        root=ROOT,
        train=True,
        download=True
    )

    test_data = datasets.MNIST(
        root=ROOT,
        train=False,
        download=True
    )

    print(train_data.classes)

    images = [image for image, label in [train_data[i] for i in range(N_IMAGES)]]
    # plot_images(images)
    # plt.show()

    n_train_samples = int(len(train_data) * VALID_RATIO)
    n_valid_samples = len(train_data) - n_train_samples

    train_data, val_data = data.random_split(
        train_data,
        [n_train_samples, n_valid_samples]
    )

    mean, std = compute_mean_and_std_on_dataset(train_data.dataset)

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = test_transforms

    train_dataloader = data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    valid_dataloader = data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE
    )

    print(f'train batch size: {next(iter(train_dataloader))[0].shape}')

    inputs, labels = next(iter(train_dataloader))
    print(f'inputs of a train batch: {inputs.shape}')
    print(f'labels of a train batch: {labels.shape}')

    num_classes = len(train_data.dataset.classes)
    lenet_model = LeNetClassifier(num_classes=num_classes)
    lenet_model = lenet_model.to(device())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters())

    num_epochs = 10
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(lenet_model.state_dict(), save_model + '/lenet_model.pt')

        # Print loss, acc end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
            "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
            )
        )
        print("-" * 59)

        # Load best model
        lenet_model.load_state_dict(torch.load(save_model + '/lenet_model.pt', weights_only=True))
        lenet_model.eval()

    plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses)

    test_data.transform = test_transforms
    test_dataloader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE
    )

    test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader)
    print(test_acc, test_loss)

    N_FILTERRED_IMAGES = 5

    images = [image for image, label in [test_data[i] for i in range(N_FILTERRED_IMAGES)]]
    filters = lenet_model.conv1.weight.data

    plot_filtered_images(images, filters)
