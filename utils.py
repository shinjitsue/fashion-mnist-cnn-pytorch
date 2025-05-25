import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import CLASS_LABELS

def get_data_loaders(data_dir, batch_size, num_workers=0, apply_augmentation=True):
    """
    Create and return data loaders for training and testing.
    
    Args:
        data_dir (str): Directory to store the datasets
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of workers for data loading
        apply_augmentation (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define the transformations for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Define transformations for training data (with augmentation)
    if apply_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and load the training data
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load the test data
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
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
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'{CLASS_LABELS[labels[i]]}')
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
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, test_loader, device):
    """
    Plot confusion matrix for the model predictions.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for the test dataset
        device: Device to run the model on
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm