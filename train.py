import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from model import FashionCNN, BaselineModel, AlternativeCNN
from utils import get_data_loaders, plot_training_history
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS,
    SAVE_DIR, MAIN_MODEL_NAME, BASELINE_MODEL_NAME, 
    ALTERNATIVE_MODEL_NAME, NO_AUG_MODEL_NAME, NO_DROPOUT_MODEL_NAME
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
    else:  # alternative
        model = AlternativeCNN(NUM_CLASSES, use_dropout=use_dropout)
        
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
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Initialize progress bar for training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
        
        for batch_idx, (data, target) in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_accuracy = validate(model, test_loader, device, epoch)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Early stopping and model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        else:
            early_stopping_counter += 1
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
    
    # Save the final model
    if model_type == "cnn":
        if use_augmentation and use_dropout:
            model_filename = MAIN_MODEL_NAME
        elif not use_augmentation:
            model_filename = NO_AUG_MODEL_NAME
        else:  # no dropout
            model_filename = NO_DROPOUT_MODEL_NAME
    elif model_type == "baseline":
        model_filename = BASELINE_MODEL_NAME
    else:  # alternative
        model_filename = ALTERNATIVE_MODEL_NAME
        
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
        for batch_idx, (data, target) in val_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            current_acc = 100 * correct / total
            val_pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
    
    accuracy = 100 * correct / total
    return accuracy

def train_with_different_lr(model_type="cnn", learning_rates=[0.01, 0.001, 0.0001]):
    """Train models with different learning rates for comparison."""
    print("Training with different learning rates...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    results = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Create model
        if model_type == "cnn":
            model = FashionCNN(NUM_CLASSES)
        elif model_type == "baseline":
            model = BaselineModel(NUM_CLASSES)
        else:
            model = AlternativeCNN(NUM_CLASSES)
            
        model = model.to(device)
        
        # Define loss function and optimizer with specific learning rate
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_accuracies = []
        
        # Shorter training for LR study
        epochs = 5
        
        for epoch in range(epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_accuracy = validate(model, test_loader, device, epoch)
            val_accuracies.append(val_accuracy)
        
        results[lr] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': val_accuracies[-1]
        }
    
    # Plot learning rate comparison
    plt.figure(figsize=(15, 5))
    
    # Plot training losses
    plt.subplot(1, 3, 1)
    for lr in learning_rates:
        plt.plot(results[lr]['train_losses'], label=f'LR={lr}')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracies
    plt.subplot(1, 3, 2)
    for lr in learning_rates:
        plt.plot(results[lr]['val_accuracies'], label=f'LR={lr}')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot final accuracies
    plt.subplot(1, 3, 3)
    final_accs = [results[lr]['final_accuracy'] for lr in learning_rates]
    plt.bar([str(lr) for lr in learning_rates], final_accs)
    plt.title('Final Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Train main model
    train(model_type="cnn", use_augmentation=True, use_dropout=True)