import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import FashionCNN, BaselineModel, AlternativeCNN
from utils import get_data_loaders, plot_training_history
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS,
    SAVE_DIR, MODEL_NAME
)

def train(model_type="cnn", use_augmentation=True, use_dropout=True):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Get data loaders with/without augmentation for ablation study
    train_loader, test_loader = get_data_loaders(
        DATA_DIR, BATCH_SIZE, NUM_WORKERS, apply_augmentation=use_augmentation
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Create the appropriate model
    if model_type == "cnn":
        model = FashionCNN(NUM_CLASSES, use_dropout=use_dropout)
    elif model_type == "baseline":
        model = BaselineModel(NUM_CLASSES)
    else:
        model = AlternativeCNN(NUM_CLASSES)
        
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    
    # Early stopping parameters
    best_val_accuracy = 0
    early_stopping_counter = 0
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    
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
        val_accuracy = validate(model, test_loader, device, epoch)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_accuracy)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_{MODEL_NAME}"))
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"best_{MODEL_NAME}")))
    
    # Save the final model
    model_filename = f"{model_type}{'_aug' if use_augmentation else ''}{'_dropout' if use_dropout else ''}_{MODEL_NAME}"
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_filename))
    print(f"Model saved to {os.path.join(SAVE_DIR, model_filename)}")
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    return model, train_losses, val_accuracies

def validate(model, test_loader, device, epoch):
    """Validate the model on the test set and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    # Initialize progress bar for validation
    val_pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                    desc=f"Epoch {epoch+1} [Validation]")
    
    with torch.no_grad():
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
    
    # Calculate and print validation accuracy
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    return val_accuracy

def train_with_different_lr(model_type="cnn", learning_rates=[0.01, 0.001, 0.0001]):
    """Train the model with different learning rates to compare performance."""
    results = []
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Get data loaders
        train_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
        
        # Create model
        model = FashionCNN(NUM_CLASSES)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train for a fixed number of epochs
        test_epochs = 5  # Use a smaller number for quick comparison
        train_losses = []
        val_accuracies = []
        
        for epoch in range(test_epochs):
            # Training
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            
            # Validation
            val_accuracy = validate(model, test_loader, device, epoch)
            val_accuracies.append(val_accuracy)
        
        # Store results
        results.append({
            'learning_rate': lr,
            'final_accuracy': val_accuracies[-1],
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        })
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    for res in results:
        plt.plot(range(1, test_epochs+1), res['train_losses'], 
                 label=f"LR: {res['learning_rate']}")
    plt.title('Training Loss by Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    for res in results:
        plt.plot(range(1, test_epochs+1), res['val_accuracies'], 
                 label=f"LR: {res['learning_rate']}")
    plt.title('Validation Accuracy by Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results



if __name__ == "__main__":
    train()