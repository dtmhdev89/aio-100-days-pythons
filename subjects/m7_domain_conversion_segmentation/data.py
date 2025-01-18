from common_libs import transforms, DataLoader, torch, plt
from torchvision.datasets import OxfordIIITPet
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# Label preprocessing
# # Closure
def create_target_transform(img_size):
    def target_transform(target):
        img = transforms.Resize(img_size)(target)  # PIL 
        img = transforms.functional.pil_to_tensor(img).squeeze_()
        img = img - 1
        img = img.to(torch.long)

        return img

    return target_transform


class TargetTransform:
    def __init__(self, img_size):
        """
        Custom target transform to resize, convert to tensor, and adjust labels.

        Args:
            img_size (tuple): The target size to resize the labels.
        """
        self.resize = transforms.Resize(img_size)

    def __call__(self, target):
        target = self.resize(target)
        target = transforms.functional.pil_to_tensor(target).squeeze_()
        target = target - 1
        target = target.to(torch.long)

        return target


class PetDataLoader():
    def __init__(self, img_size) -> None:
        self.__img_size = img_size
        self.transform = self.data_transform()
        self.target_transform = TargetTransform(img_size=img_size)

    def data_transform(self):
        transform = transforms.Compose([
              transforms.Resize(self.__img_size),
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))
        ])

        return transform

    def get_dataset(self, split: Literal["trainval", "test"]):
        train_set = OxfordIIITPet(
            root="pets_data",
            split=split,
            target_types="segmentation",
            transform=self.transform,
            target_transform=self.target_transform,
            download=True
        )

        return train_set

if __name__ == "__main__":
    img_size = (128, 128)
    num_classes = 3

    transform = transforms.Compose([            # PIL
                transforms.Resize(img_size),  # PIL
                transforms.ToTensor(),        # 1) Tensor      2) [0, 1]
                # transforms.Normalize((0.485, 0.456, 0.406),
                #                     (0.229, 0.224, 0.225))
    ])

    target_transform = create_target_transform(img_size=img_size)

    train_set = OxfordIIITPet(
        root="pets_data",
        split="trainval",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    sample = train_set[0]

    print(sample[1].min())
    print(sample[1].shape)

    batch_size = 64
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # Get a batch of training data
    images, masks = next(iter(train_loader))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # Plot 8 pairs of images and their masks
    for idx in range(8):
        row = idx // 2
        col = (idx % 2) * 2

        # Plot original image
        axes[row, col].imshow(images[idx].permute(1,2,0))
        axes[row, col].axis('off')
        axes[row, col].set_title('Image')

        # Plot segmentation mask
        axes[row, col+1].imshow(masks[idx], cmap='gray')
        axes[row, col+1].axis('off')
        axes[row, col+1].set_title('Mask')

    plt.tight_layout()
    plt.show()
