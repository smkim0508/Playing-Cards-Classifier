# training the model and validating results
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

# training progress track
# NOTE: import from tqdm.notebook for Colab/Jupyter
from tqdm import tqdm

# defined model
from model import CardsClassifier
# defined dataset class
from dataset import PlayingCardDataset

# instantiate the model
model = CardsClassifier(num_classes=53)

# define loss function
# NOTE: criterion is the typical name for loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# set up datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor()
    # NOTE: can add any other transforms here
])

# download the latest version of the playing cards dataset and cache it locally
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
print(f"Path to downloaded Kaggle dataset files: {path}")

train_path = path + "/train"
test_path = path + "/test"
validation_path = path + "/valid"

train_dataset = PlayingCardDataset(data_dir=train_path, transform=transform)
test_dataset = PlayingCardDataset(data_dir=test_path, transform=transform)
validation_dataset = PlayingCardDataset(data_dir=validation_path, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)

# use GPU if available
cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
print(f"Using device: {device}")
if cuda_available:
    model.to(device) # forward model to GPU if available

# set up the training loop
num_epochs = 5 # small training loop for now
# define empty lists to store loss values
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train() # ensure model is in training mode
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training Loop"):
        if cuda_available:
            images, labels = images.to(device), labels.to(device) # forward images and labels to GPU, if available because tensors must match device
        optimizer.zero_grad() # reset gradient
        outputs = model(images) # forward pass images in batches defined by loader
        loss = criterion(outputs, labels) # calculate loss
        loss.backward() # back prop loss, handled by PyTorch internally
        optimizer.step() # update weights
        running_loss = loss.item() * images.size(0) # normalize by multiplying by num of images per batch and later dividing by total num of images
    train_loss = running_loss / len(train_dataset) # train loss for an epoch becomes the average loss
    train_losses.append(train_loss)

    # validation
    model.eval() # ensure model is in validation mode
    running_loss = 0.0
    # ensure model weights aren't updated during this phase
    with torch.no_grad():
        for images, labels in tqdm(validation_loader, desc="Validation Loop"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss = loss.item() * images.size(0)
        val_loss = running_loss / len(validation_dataset)
        val_losses.append(val_loss)

    # print loss every epoch for monitoring
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
