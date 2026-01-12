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

# Dataset Creation
- By creating the dataset using Torch's Dataset base class and wrapping it with DataLoader, we can utilize PyTorch to automatically load in data in parallel
- Dataset is an iterable type
- Dataloader also helps to batch the samples and enable shuffling

### TODO:
- finish exploring the playing cards data
- build classifier model
- train