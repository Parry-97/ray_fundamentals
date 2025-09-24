# Torchvision Transforms in ray_tune.py

## Overview
This document explains how `torchvision` transforms are defined and applied for preprocessing MNIST dataset images in the `ray_tune.py` file.

## Transform Pipeline

### Import Required Components
```python
from torchvision.transforms import Compose, Normalize, ToTensor
```

### Transform Definition
```python
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
```

The pipeline consists of two sequential operations:

#### 1. ToTensor()
- **Converts**: PIL images → PyTorch tensors
- **Scaling**: Pixel values from [0, 255] → [0.0, 1.0]
- **Dimension reordering**: (H, W, C) → (C, H, W) for PyTorch compatibility

#### 2. Normalize((0.5,), (0.5,))
- **Formula**: `(pixel - mean) / std`
- **Parameters**: 
  - Mean = 0.5 (single value for grayscale)
  - Std = 0.5 (single value for grayscale)
- **Result**: Transforms values from [0, 1] → [-1, 1]
  - Min: (0 - 0.5) / 0.5 = -1
  - Max: (1 - 0.5) / 0.5 = 1

### Normalization Axis Details

#### Per-Channel Normalization
The `Normalize` transform applies normalization **per channel** along the channel axis (dimension 0):

- **Tensor Shape**: After `ToTensor()`, images have shape `(C, H, W)`
  - C = channels (1 for MNIST grayscale)
  - H = height (28 pixels)
  - W = width (28 pixels)

#### How It Works
- **Application**: Each pixel in a channel is normalized using that channel's mean and std
- **MNIST Example**: With shape `(1, 28, 28)`:
  - All 784 pixels (28×28) use the same mean (0.5) and std (0.5)
  - Operation: `normalized_pixel = (original_pixel - 0.5) / 0.5`
  - Applied element-wise to every pixel in the spatial dimensions

#### What Normalization Does NOT Do
- ❌ Does NOT compute statistics across images in a batch
- ❌ Does NOT compute mean/std from the spatial dimensions
- ❌ Does NOT normalize across channels (each channel has independent stats)

#### What Normalization DOES Do
- ✅ Applies pre-defined mean/std per channel
- ✅ Normalizes every pixel within each channel independently
- ✅ Operates element-wise across spatial dimensions (H, W)

#### RGB Image Comparison
For RGB images with shape `(3, H, W)`, you would use:
```python
# ImageNet statistics example
Normalize((0.485, 0.456, 0.406),  # per-channel means (R, G, B)
          (0.229, 0.224, 0.225))   # per-channel stds (R, G, B)
```
Each channel is normalized with its own mean and standard deviation.

## Application

### Dataset Integration
```python
dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,  # Transforms applied here
)
```

### Execution Timing
- **Lazy evaluation**: Transforms are applied on-the-fly
- **Triggered when**:
  - Accessing dataset items: `dataset[i]`
  - DataLoader iteration during training
- **Memory efficient**: No storage of transformed images

## Data Flow Summary

| Stage | Data Format | Value Range |
|-------|------------|-------------|
| Raw MNIST | 28×28 grayscale PIL | [0, 255] |
| After ToTensor() | 1×28×28 tensor | [0, 1] |
| After Normalize() | 1×28×28 tensor | [-1, 1] |

## Benefits

1. **Standardized Input**: Normalized data improves neural network training
2. **PyTorch Compatibility**: Tensor format expected by models
3. **Numerical Stability**: Zero-centered data enhances gradient flow
4. **Memory Efficiency**: On-demand transformation reduces memory usage

## Key Takeaway
The transformation pipeline efficiently preprocesses MNIST images from their original format into normalized tensors optimized for deep learning training, applying transformations lazily to conserve memory.