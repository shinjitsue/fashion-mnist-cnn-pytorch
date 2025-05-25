import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import FashionCNN
from utils import get_data_loaders, plot_training_history
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS,
    SAVE_DIR, MODEL_NAME
)

def train():
    # Set device (CPU in this case)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Create the model
    model = FashionCNN(NUM_CLASSES)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Initialize progress bar
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        
        # Training
        for i, (images, labels) in train_pbar:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': running_loss / (i + 1)})
        
        # Calculate average training loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        
        # Initialize progress bar for validation
        val_pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                        desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        
        with torch.no_grad():  # Disable gradient calculation for validation
            for i, (images, labels) in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # Update statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'accuracy': 100 * correct / total})
        
        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Training Loss: {epoch_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the model
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}_{MODEL_NAME}"))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
    print(f"Model saved to {os.path.join(SAVE_DIR, MODEL_NAME)}")
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    return model

if __name__ == "__main__":
    train()