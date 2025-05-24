"""
Utility functions for the Fashion MNIST CNN project.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import CLASS_LABELS

def get_data_loaders(data_dir, batch_size, num_workers=0):
    """
    Create and return data loaders for training and testing.
    
    Args:
        data_dir (str): Directory to store the datasets
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
    ])
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and load the training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load the test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def plot_sample_images(data_loader):
    """
    Plot sample images from the dataset with their labels.
    
    Args:
        data_loader: Data loader for the dataset
    """
    # Get a batch of training data
    examples = iter(data_loader)
    images, labels = next(examples)
    
    # Plot images in a grid
    plt.figure(figsize=(10, 10))
    for i in range(25):  # Display 25 images in a 5x5 grid
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(CLASS_LABELS[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, val_accuracies):
    """
    Plot training loss and validation accuracy.
    
    Args:
        train_losses (list): Training losses for each epoch
        val_accuracies (list): Validation accuracies for each epoch
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_loader, device):
    """
    Plot confusion matrix for the trained model.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for the test dataset
        device: Device to run the model on
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store true labels and predictions
    y_true = []
    y_pred = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Append batch prediction results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_LABELS, 
                yticklabels=CLASS_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()