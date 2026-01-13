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

### TODO:
- build classifier model
- train