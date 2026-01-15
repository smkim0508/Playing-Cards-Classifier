# testing the trained model
# NOTE: loads in a trained model from a .pth file, saved under checkpoints/

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

# image related imports
from PIL import Image

def preprocess_image(image_path, transform):
    """
    Load and preprocess image, returns tuple of image and tensor.
    """
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    """
    Make prediction with model.
    Probabilities represent the confidencce for each target class.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
    """
    Intuitive visualization of predictions.
    """
    # init plots
    fig, axarr = plt.subplots(1, 2, figsize=(10, 7))
    
    # display just the original image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# run evaluation
if __name__ == "__main__":
    # get the test data path
    path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
    print(f"Path to downloaded Kaggle dataset files: {path}")
    test_path = path + "/test"
    test_image = test_path + "/queen of hearts/1.jpg"

    transform = transforms.Compose([
        transforms.Resize([128,128]),
        transforms.ToTensor()
    ])

    # build dataset
    test_dataset = PlayingCardDataset(data_dir=test_path, transform=transform)

    # load in the model from previously saved checkpoint
    checkpoint_dir = "image_classifier/checkpoints/model_2.pth"
    model = CardsClassifier(num_classes=53)
    model.load_state_dict(torch.load(checkpoint_dir))

    # use GPU if available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"Using device: {device}")
    if cuda_available:
        model.to(device) # forward model to GPU if available

    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)

    visualize_predictions(original_image, probabilities, test_dataset.classes)
