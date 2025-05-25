import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import random


from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from itertools import cycle
from model import FashionCNN, BaselineModel, AlternativeCNN
from utils import get_data_loaders, plot_confusion_matrix
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, 
    NUM_CLASSES, SAVE_DIR, MODEL_NAME, CLASS_LABELS
)

def load_model(model_file, model_class, device):
    """Load a specific model from file with error handling"""
    model_path = os.path.join(SAVE_DIR, model_file)
    model = model_class(NUM_CLASSES)
    
    try:
        # Check if model file exists
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
    except Exception as e:
        print(f"Error loading model: {str(e)}. Using untrained model.")
    
    return model.to(device)

def calculate_per_class_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score for each class"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\nPer-Class Metrics:")
    print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")
    for i in range(NUM_CLASSES):
        print(f"{CLASS_LABELS[i]}\t{precision[i]:.4f}\t\t{recall[i]:.4f}\t\t{f1[i]:.4f}\t\t{support[i]}")
    
    return precision, recall, f1, support

def plot_roc_curves(y_true, y_score):
    """Plot ROC curves for multiclass classification"""
    # Binarize the labels for one-vs-rest ROC curves
    y_true_bin = np.zeros((len(y_true), NUM_CLASSES))
    for i in range(len(y_true)):
        y_true_bin[i, y_true[i]] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(10, 8))
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 
                   'magenta', 'black', 'orange', 'brown'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{CLASS_LABELS[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def visualize_model_predictions(model, test_loader, device, num_images=10, errors_only=False):
    """
    Visualize model predictions, optionally focusing on errors only.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for the test dataset
        device: Device to run the model on
        num_images: Number of images to visualize
        errors_only: If True, show only misclassified examples
    """
    model.eval()
    
    if errors_only:
        # Collect misclassified examples
        errors_images = []
        errors_preds = []
        errors_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                # Find misclassifications
                error_mask = preds != labels
                misclassified_images = images[error_mask]
                misclassified_preds = preds[error_mask]
                misclassified_labels = labels[error_mask]
                
                # Add errors to our lists
                errors_images.extend(misclassified_images.cpu().numpy())
                errors_preds.extend(misclassified_preds.cpu().numpy())
                errors_labels.extend(misclassified_labels.cpu().numpy())
                
                if len(errors_images) >= num_images:
                    break
        
        # Take only the requested number of samples
        sample_images = errors_images[:num_images]
        predicted_labels = errors_preds[:num_images]
        sample_labels = errors_labels[:num_images]
        
        title_prefix = "Challenging Cases: Misclassified Examples"
    else:
        # Get a batch of random test images
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        # Select random images
        indices = random.sample(range(len(images)), num_images)
        sample_images_tensor = images[indices].to(device)
        sample_labels = labels[indices].tolist()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(sample_images_tensor)
            _, predicted = torch.max(outputs, 1)
        
        predicted_labels = predicted.cpu().tolist()
        sample_images = [img.cpu().numpy() for img in sample_images_tensor]
        
        title_prefix = "Model Predictions"
    
    # Display images with true and predicted labels
    rows = (num_images + 4) // 5  # Calculate rows needed (ceiling division)
    cols = min(5, num_images)     # Use at most 5 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if rows * cols == 1:  # Handle the case of a single image
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(min(num_images, len(sample_images))):
        if isinstance(sample_images[i], np.ndarray):
            img = sample_images[i][0]  # Get the grayscale channel
        else:
            img = sample_images[i]
            
        axes[i].imshow(img, cmap='gray')
        
        # Set title with true and predicted labels
        if errors_only:
            title = f"True: {CLASS_LABELS[sample_labels[i]]}\n"
            title += f"Pred: {CLASS_LABELS[predicted_labels[i]]}"
            color = 'red'  # Errors are always in red
        else:
            title = f"True: {CLASS_LABELS[sample_labels[i]]}\n"
            title += f"Pred: {CLASS_LABELS[predicted_labels[i]]}"
            color = 'green' if sample_labels[i] == predicted_labels[i] else 'red'
        
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(sample_images), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()


def plot_model_comparisons(model_metrics):
    """Plot comparison of different models"""
    models = [m[0] for m in model_metrics]
    accuracies = [m[1]['accuracy'] for m in model_metrics]
    
    plt.figure(figsize=(12, 6))
    
    # Plot comparison of accuracies
    plt.subplot(1, 2, 1)
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Plot comparison of F1 scores (average across classes)
    f1_scores = [np.mean(m[1]['f1']) for m in model_metrics]
    
    plt.subplot(1, 2, 2)
    plt.bar(models, f1_scores, color='lightgreen')
    plt.title('Model F1 Score Comparison')
    plt.ylabel('Average F1 Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def evaluate():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    # Get data loaders
    _, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    # ===== Main Model Evaluation =====
    print("\n=== Main Model Evaluation ===")
    main_model = load_model(MODEL_NAME, FashionCNN, device)
    main_metrics = evaluate_model(main_model, test_loader, device)
    
    # ===== MODEL COMPARISONS =====
    # Try to load and evaluate other models if they exist
    try:
        # Compare with baseline (logistic regression)
        print("\n=== Baseline Model (Logistic Regression) ===")
        baseline_model = load_model("baseline_fashion_mnist_cnn.pth", BaselineModel, device)
        baseline_metrics = evaluate_model(baseline_model, test_loader, device)
        
        # Compare with alternative CNN
        print("\n=== Alternative CNN Architecture ===")
        alt_model = load_model("alternative_fashion_mnist_cnn.pth", AlternativeCNN, device)
        alt_metrics = evaluate_model(alt_model, test_loader, device)
        
        # ===== ABLATION STUDIES =====
        # Compare with no augmentation model
        print("\n=== Ablation Study: No Data Augmentation ===")
        no_aug_model = load_model("cnn_dropout_fashion_mnist_cnn.pth", FashionCNN, device)
        no_aug_metrics = evaluate_model(no_aug_model, test_loader, device)
        
        # Compare with no dropout model
        print("\n=== Ablation Study: No Dropout ===")
        no_dropout_model = load_model("cnn_aug_fashion_mnist_cnn.pth", FashionCNN, device)
        no_dropout_metrics = evaluate_model(no_dropout_model, test_loader, device)
        
        # Plot comparison of models
        plot_model_comparisons([
            ("Main CNN", main_metrics), 
            ("Baseline", baseline_metrics),
            ("Alternative", alt_metrics),
            ("No Aug", no_aug_metrics),
            ("No Dropout", no_dropout_metrics)
        ])
    except Exception as e:
        print(f"Could not run full model comparison: {str(e)}")
        print("Continuing with main model evaluation only.")
    
    # Visualize some predictions from the main model
    visualize_model_predictions(main_model, test_loader, device, errors_only=False)

def evaluate_model(model, test_loader, device, plot_visuals=True):
    """
    Streamlined model evaluation focused on essential metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for the test dataset
        device: Device to run the model on
        plot_visuals: Whether to generate visualizations
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating model"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate overall accuracy
    accuracy = 100 * np.mean(all_predictions == all_targets)
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None
    )
    
    # Print per-class metrics
    if plot_visuals:
        print("\nPer-Class Metrics:")
        print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")
        for i in range(NUM_CLASSES):
            print(f"{CLASS_LABELS[i]}\t{precision[i]:.4f}\t\t{recall[i]:.4f}\t\t{f1[i]:.4f}\t\t{support[i]}")
        
        # Generate visualizations if requested
        plot_confusion_matrix(model, test_loader, device)
        plot_roc_curves(all_targets, all_probabilities)
        visualize_model_predictions(model, test_loader, device, errors_only=True)
    
    # Return metrics for comparison
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    

if __name__ == "__main__":
    evaluate()