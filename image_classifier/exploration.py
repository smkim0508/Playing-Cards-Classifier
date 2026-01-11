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

# for image display
from PIL import Image, ImageTk
import tkinter as tk

import os

print(f"Starting Exploration with Playing Cards Dataset!")
print(f"PyTorch version: {torch.__version__}\nTorchvision version: {torchvision.__version__}\ntimm version: {timm.__version__}\nNumPy version: {np.__version__}\nPandas version: {pd.__version__}")

# Download the latest version of the playing cards dataset and cache it locally
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

train_path = path + "/train"
test_path = path + "/test"
validation_path = path + "/valid"

dataset = PlayingCardDataset(
    data_dir=train_path
)

print(f"Length of dataset: {len(dataset)}")
idx = 6000
print(f"Getting random item from idx {idx}: {dataset[idx]}")

image, label = dataset[idx]

# display the image on a pop up window
# NOTE: this is usually handled natively by Notebooks like Colab
root = tk.Tk()
root.title(f"Playing Card {idx}: {label}")
tk_image = ImageTk.PhotoImage(image)
tk_label = tk.Label(root, image=tk_image)
tk_label.pack()
root.mainloop()

# print the label for image from dataset
print(f"Label: {label}")

# try to map values with folder labels
target_to_class = {v: k for k, v in dataset.data.class_to_idx.items()}
print(f"target to class: {target_to_class}")