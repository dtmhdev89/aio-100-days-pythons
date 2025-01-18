import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch import nn
import math
import os
import sys

up_levels = [".."] * 3
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    *up_levels
))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.google_drive_util import gdrive_download_and_unzip

if __name__ == "__main__":
    zip_path = gdrive_download_and_unzip(id='1vSevps_hV5zhVf6aWuN8X7dd-qSAIgcc',
                                         cached_download=True)
    print(zip_path)
