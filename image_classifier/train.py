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

# defined model
from image_classifier.model import CardsClassifier
# defined dataset class
from image_classifier.dataset import PlayingCardDataset

# TODO: define training loop
