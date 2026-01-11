# dataset and pytorch exploration
import torch
import torch.nn as nn # NN specific library
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm # image classification-specific architecture library
from torch.utils.data import DataLoader, Dataset
import kagglehub

import matplotlib.pyplot as plt # data visualization
import pandas as pd
import numpy as np

import os

print(f"Starting Exploration!")
print(f"PyTorch version: {torch.__version__}\nTorchvision version: {torchvision.__version__}\ntimm version: {timm.__version__}\nNumPy version: {np.__version__}\nPandas version: {pd.__version__}")

# Download latest version
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")

print(f"Path to downloaded Kaggle dataset files: {path}")

class PlayingCardDataset(Dataset):
    """
    Playing Card Dataset class that inherits from PyTorch's base Dataset.
    Currently attempting to set this up using kaggle dataset.
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(root=data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

data_dir = path

dataset = PlayingCardDataset(
    data_dir=data_dir
)

print(f"Length of dataset: {len(dataset)}")
