# PyTorch-Practice
Refining my PyTorch skills to build models from scratch

Various projects exploring different model architectures.
This README will also contain my personal notes as I review key concepts and practice with examples.

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
- PyTorch's loss function provides .backward() that automatically updates model weights w/ back propagation
- The main training for loop should run for n_epoch times, and each loop exhaust the entire training data in batches using previously defined dataloader
    - Instantiate dataloader for training, test, and validation data
    - There are methods available to split dataset into train, test, validation, typically like 70/20/10, but can use pre-defined dataset (e.g. Playing Cards Dataset used in this practice).
- Optimizer actually decides the direction of gradient descent based on the loss.
    - PyTorch provides useful optimizers like Adam
    - *Observe how to define my own optimizer, and if PyTorch supports this*
- Learning rate determines the step size, and tools like learning rate scheduler help adjust LR as training goes on (based on loss threshold or epochs)
### TODO:
- build classifier model
- train