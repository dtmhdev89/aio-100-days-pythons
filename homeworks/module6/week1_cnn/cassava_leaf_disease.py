import os
from imutils import paths
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


def show_labels(data_paths):
    fig, ax = plt.subplots(1, len(data_paths), figsize=(12,6))
    for idx, (key, sub_dir) in enumerate(data_paths.items()):
        labels = os.listdir(sub_dir)
        list_data = []
        for label in labels:
            image_files = list(paths.list_images(os.path.join(sub_dir, label)))
            list_data.append(len(image_files))
        ax[idx].bar(labels, list_data)
        ax[idx].set_title(key)
    plt.tight_layout()
    plt.show()


def plot_images(data_dir, label, num_sample=6):
    data_dir = os.path.join(data_dir, label)
    image_files = list(paths.list_images(data_dir))[:num_sample]
    fig, ax = plt.subplots(2,num_sample//2, figsize=(14,7))
    for i, image_dir in enumerate(image_files):
        img = Image.open(image_dir)
        label = image_dir.split('/')[-2]
        ax[i//(num_sample//2)][i%(num_sample//2)].imshow(img)
        ax[i//(num_sample//2)][i%(num_sample//2)].set_title(labels_dict[label])
        ax[i//(num_sample//2)][i%(num_sample//2)].axis('off')
    plt.tight_layout()
    plt.show()


def loader(path):
    return Image.open(path)


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6,
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
        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
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
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()


def load_model(model_path, num_classes=5):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True))
    lenet_model.eval()
    return lenet_model


def inference(img_path, model):
    image = Image.open(img_path)
    img_size = 150

    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img_new = img_transform(image)
    img_new = torch.unsqueeze(img_new, 0)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item(), yhat.item()


if __name__ == "__main__":
    base_data_path = os.path.join('cassavaleafdata')
    data_paths = {
        'train': os.path.join(base_data_path, 'train'),
        'valid': os.path.join(base_data_path, 'validation'),
        'test': os.path.join(base_data_path, 'test')
    }

    labels_dict = {
        "cbb": "Cassava Bacterial Blight (CBB)",
        "cbsd": "Cassava Brown Streak Disease (CBSD)",
        "cgm": "Cassava Green Mottle (CGM)",
        "cmd": "Cassava Mosaic Disease (CMD)",
        "healthy": "Healthy"
    }

    plot_images(data_paths['train'], label="cbb")
    plot_images(data_paths['train'], label="cgm")
    plot_images(data_paths['train'], label="cmd")
    plot_images(data_paths['train'], label="healthy")

    img_size = 150

    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(
        root=data_paths['train'],
        loader=loader,
        transform=train_transforms
    )
    val_data = datasets.ImageFolder(
        root=data_paths['valid'],
        transform=train_transforms
    )
    test_data = datasets.ImageFolder(
        root=data_paths['test'],
        transform=train_transforms
    )

    BATCH_SIZE = 512

    train_dataloader = data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    valid_dataloader = data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE
    )

    test_dataloader = data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE
    )

    num_classes = len(train_data.classes)
    lenet_model = LeNetClassifier(num_classes=num_classes)
    lenet_model = lenet_model.to(device())

    criterion = nn.CrossEntropyLoss()
    lr = 2e-4
    optimizer = optim.Adam(lenet_model.parameters(), lr=lr)

    num_epochs = 10
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)

    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100

    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()

        train_acc, train_loss = train(lenet_model, optimizer, criterion,
                                      train_dataloader, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        eval_acc, eval_loss = evaluate(lenet_model, criterion,
                                       valid_dataloader)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)

        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(lenet_model.state_dict(),
                       save_model + '/lenet_model_cassava.pt')

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
        lenet_model.load_state_dict(
            torch.load(save_model + '/lenet_model_cassava.pt',
                       weights_only=True)
        )
        lenet_model.eval()

    plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses)

    test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader)
    print(test_acc, test_loss)

    model = load_model(os.path.join('model/lenet_model_cassava.pt'))
    preds = inference(os.path.join('cassavaleafdata/test/cbsd/test-cbsd-1.jpg'), model)
    print(preds)
