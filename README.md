# Playing-Cards-Classifier
Refining my PyTorch skills to build models from scratch - Playing Cards Image Classifier using EfficientNet as base.
*This README will also contain my personal notes as I review key concepts and practice with examples.*

## How to Run
1. Download the dataset using the Kaggle API (see below for link)
2. To train your own model, modify model architecture in `model.py` (optional) and run the training loop in `train.py` with any implementation detail modifications.
3. Alternatively, use the pre-trained EfficientNet model under `checkpoints/`.
4. Test the outputs using randomized test images with the help of `evaluation.py`.

## Evaluation
TODO

## NOTES
### Fundamentals
PyTorch usage can be mainly characterized by 3 main components:
1. PyTorch Dataset
2. PyTorch Model
3. PyTorch Training Loop

Additionally, evaluation, testing, and validation comes after training.

### Dataset Creation
- By creating the dataset using Torch's Dataset base class and wrapping it with DataLoader, we can utilize PyTorch to automatically load in data in parallel
- Dataset is an iterable type
- Dataloader also helps to batch the samples and enable shuffling
    - Batch size should be a power of 2.

### PyTorch Models
- Can be created layer-by-layer from scratch, or import SOTA models and modify
- Timm provides some SOTA image classification models
- Key is understanding the shape of each layer
- Typically, when importing model, the last layer is most important since it directly correlates to the desired targets.
- PyTorch models inherit from the torch.nn.Module base class
    - under init for the children class, use super().__init__ to init with parent class too
- Typically contains __init__() and forward() methods as baseline
    - init defines the different components of this model
    - forward connects the parts in a forward pass
- Sequential() can be used to connect features together
- For classification tasks, a Linear layer (nn.Linear) is typically used to map the output of base model to number of target classes
- If using basemodel, the final feature is cut out since Linear layer exists

### Training the Model
- The key idea is simple, same as any NN training: use loss function to evaluate accuracy and use back propagation to update model weight based on loss as training continues
- PyTorch's loss function provides .backward() that automatically finds gradients for all parameters in layers w.r.t. loss.
    - Loss is defined but never explicitly given information about the model parameters; PyTorch internally computes gradients w.r.t. to each parameters.
- The main training for loop should run for n_epoch times, and each loop exhaust the entire training data in batches using previously defined dataloader
    - Instantiate dataloader for training, test, and validation data
    - There are methods available to split dataset into train, test, validation, typically like 70/20/10, but can use pre-defined dataset (e.g. Playing Cards Dataset used in this practice).
- Optimizer actually decides the direction of gradient descent based on the loss/gradients and updates parameter weights.
    - PyTorch provides useful optimizers like Adam
    - *Observe how to define my own optimizer, and if PyTorch supports this*
- Learning rate determines the step size, and tools like learning rate scheduler help adjust LR as training goes on (based on loss threshold or epochs)
- PyTorch's nn.Module base class offers parameters() that returns list of params of the built model (automatically scans)
- Loss is tracked in a simple list typically.
- Tracking the training progress: for this, tqdm library can be used to show a live progress bar.
    - Simply wrap the data loaders w/ adequate descriptions.
- Save checkpoints of trained model using torch.save()

### Evaluating the Model
- Load in saved versions of the model (or use the trained instances if on Colab/Jutyper Notebook)
- Pass images through the trained model, and convert raw classification output to desired format
    - e.g. probabilities corresponding to each target class using softmax() or take the highest confidence result with argmax()

## Acknowledgements
The Playing Cards Dataset from Kaggle is used to train this image classifier: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data.
- This data is downloaded using the Kaggle API, please see example in `exploration.py` for further reference.

Following Rob Mulla's guide on [YouTube](!https://www.youtube.com/watch?v=tHL5STNJKag) and modified for self-learning.