# Fashion-MNIST CNN Classification Project

A comprehensive deep learning project implementing and comparing multiple Convolutional Neural Network architectures for Fashion-MNIST classification using PyTorch.

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline including data preprocessing, model design, training, evaluation, and analysis. It implements multiple CNN architectures with systematic ablation studies to understand the impact of various design choices.

## 📊 Performance Results

### Model Comparison Summary

| Model | Accuracy | F1-Score | Key Features |
|-------|----------|----------|--------------|
| **CNN (No Augmentation)** | **92.71%** | **0.9273** | Best overall performance |
| CNN (No Dropout) | 90.80% | 0.9072 | Strong generalization |
| Main CNN (Aug + Dropout) | 89.95% | 0.8980 | Balanced approach |
| Alternative CNN | 88.31% | 0.8822 | Simpler architecture |
| Baseline (Logistic Regression) | 66.98% | 0.6647 | Simple baseline |

### Key Scientific Findings

1. **Data Augmentation Impact**: Surprisingly, the model without augmentation achieved the highest accuracy (92.71% vs 89.95%), suggesting that for Fashion-MNIST:
   - The dataset already contains sufficient variation
   - Aggressive augmentation may introduce noise rather than helpful diversity
   - This finding challenges conventional wisdom about always using data augmentation

2. **Regularization Trade-off**: The no-dropout model (90.80%) outperformed the model with dropout (89.95%), indicating:
   - The model capacity is well-suited for the dataset complexity
   - Dropout might be too aggressive for this particular architecture
   - The model shows good generalization without explicit regularization

3. **Architecture Effectiveness**: The main CNN consistently outperforms the alternative shallow architecture, validating the design decisions for deeper networks with batch normalization.

## 🏗️ Architecture Design

### Main CNN Architecture

````python
class FashionCNN(nn.Module):
    """
    3-layer CNN with batch normalization and dropout
    - Conv1: 1→32 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool
    - Conv2: 32→64 channels, 3x3 kernel, BatchNorm, ReLU, MaxPool  
    - Conv3: 64→128 channels, 3x3 kernel, BatchNorm, ReLU
    - FC1: 6272→256, ReLU, Dropout
    - FC2: 256→10 (output)
    """
````
