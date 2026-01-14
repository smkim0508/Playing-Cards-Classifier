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

# dataset class to explore
from image_classifier.dataset import PlayingCardDataset

# for image display
from PIL import Image, ImageTk
import tkinter as tk

import os

print(f"Starting Exploration with Playing Cards Dataset!")
print(f"PyTorch version: {torch.__version__}\nTorchvision version: {torchvision.__version__}\ntimm version: {timm.__version__}\nNumPy version: {np.__version__}\nPandas version: {pd.__version__}")

# Download the latest version of the playing cards dataset and cache it locally
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
print(f"Path to downloaded Kaggle dataset files: {path}")

train_path = path + "/train"
test_path = path + "/test"
validation_path = path + "/valid"

# dataset = PlayingCardDataset(
#     data_dir=train_path
# )

# print(f"Length of dataset: {len(dataset)}")
idx = 6000
# print(f"Getting random item from idx {idx}: {dataset[idx]}")

# image, label = dataset[idx]

# # display the image on a pop up window
# # NOTE: this is usually handled natively by Notebooks like Colab
# root = tk.Tk()
# root.title(f"Playing Card {idx}: {label}")
# tk_image = ImageTk.PhotoImage(image)
# tk_label = tk.Label(root, image=tk_image)
# tk_label.pack()
# root.mainloop()

# # print the label for image from dataset
# print(f"Label: {label}")

# # try to map values with folder labels
# target_to_class = {v: k for k, v in dataset.data.class_to_idx.items()}
# print(f"target to class: {target_to_class}")

# now, re-do dataset with resized images and converted to tensor to ensure uniform format
transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor()
    # NOTE: can add any other transforms here
])

dataset_transformed = PlayingCardDataset(
    data_dir=train_path,
    transform=transform
)

# NOTE: now image is no longer a PIL image, but a torch tensor
image, label = dataset_transformed[idx]
print(f"Image shape for idx {idx}: {image.shape}, label: {label}")

# also exploring the shape of the image/label tensors
for image, label in dataset_transformed:
    break # just fetch the first image and label
print(f"Iterating DATASET, shapes of first item's image: {image.shape}, label: {label}")

# use a Dataloader to wrap the dataset
dataloader = DataLoader(
    dataset=dataset_transformed,
    batch_size=32,
    shuffle=True # used for training mostly
)

# now compare the shapes of image/label tensors after wrapping it in dataloader
# NOTE: now, images and labels are batched; labels are also randomized since shuffled, and each of the 32 labels correspond to each image in batch.
# for example, if shuffle = False, we'd expect all labels in the first batch to be [0...0]
for images, labels in dataloader:
    break # just fetch the first images/labels as tensors
print(f"Iterating DATALOADER, shapes of first batch of images shape: {images.shape}, labels shape: {labels.shape}, labels: {labels}")
print(f"Additionally, images example: {images}") # comment out to avoid cluttering output