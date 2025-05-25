import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from model import FashionCNN
from utils import get_data_loaders, plot_confusion_matrix
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, 
    NUM_CLASSES, SAVE_DIR, MODEL_NAME, CLASS_LABELS
)

def evaluate():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    # Get data loaders
    _, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    # Load the trained model
    model_path = os.path.join(SAVE_DIR, MODEL_NAME)
    model = FashionCNN(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Calculate accuracy on the test set
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, device)
    
    # Visualize some predictions
    visualize_predictions(model, test_loader, device)

def visualize_predictions(model, test_loader, device, num_images=10):
    """
    Visualize model predictions on random test images.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for the test dataset
        device: Device to run the model on
        num_images: Number of random images to visualize
    """
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Select random images
    indices = random.sample(range(len(images)), num_images)
    sample_images = images[indices].to(device)
    sample_labels = labels[indices].tolist()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(sample_images)
        _, predicted = torch.max(outputs, 1)
    
    predicted_labels = predicted.cpu().tolist()
    
    # Display images with true and predicted labels
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = sample_images[i].cpu().numpy()[0]  # Get the grayscale channel
        axes[i].imshow(img, cmap='gray')
        
        # Set title with true and predicted labels
        title = f"True: {CLASS_LABELS[sample_labels[i]]}\n"
        title += f"Pred: {CLASS_LABELS[predicted_labels[i]]}"
        
        # Highlight correct/incorrect predictions
        color = 'green' if sample_labels[i] == predicted_labels[i] else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()