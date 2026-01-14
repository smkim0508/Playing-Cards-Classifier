# creating the baseline PyTorch model w/ pre-trained SOTA models as base
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

class CardsClassifier(nn.Module):
    """
    Simple card classification model, inherits from nn.Module base.
    """
    def __init__(self, num_classes: int = 53):
        """
        Defines components of this model.
        Num classes set to 53 (number of playing cards).
        """
        # define base model using timm's pretrained model
        pass

    def forward(self, x):
        """
        Forward pass that connects components of this model
        """
        pass
