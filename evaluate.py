import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from itertools import cycle
from model import FashionCNN, BaselineModel, AlternativeCNN
from utils import get_data_loaders, plot_confusion_matrix
from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, 
    NUM_CLASSES, SAVE_DIR, CLASS_LABELS,
    MAIN_MODEL_NAME, BASELINE_MODEL_NAME, ALTERNATIVE_MODEL_NAME,
    NO_AUG_MODEL_NAME, NO_DROPOUT_MODEL_NAME
)

def load_model(model_file, model_class, device):
    """Load a specific model from file with error handling"""
    model_path = os.path.join(SAVE_DIR, model_file)
    model = model_class(NUM_CLASSES)
    
    if not os.path.exists(model_path):
        print(f"Warning: {model_file} not found. Skipping...")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded {model_file}")
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        return None
    
    return model.to(device)

def calculate_per_class_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score for each class"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    print("\nPer-Class Metrics:")
    print("Class\t\tPrecision\tRecall\t\tF1-Score\tSupport")
    for i in range(NUM_CLASSES):
        print(f"{CLASS_LABELS[i]:<12}\t{precision[i]:.4f}\t\t{recall[i]:.4f}\t\t{f1[i]:.4f}\t\t{support[i]}")
    
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
        images_to_show = []
        labels_to_show = []
        predictions_to_show = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # Find misclassified examples
                incorrect_mask = predicted != labels
                incorrect_images = images[incorrect_mask]
                incorrect_labels = labels[incorrect_mask]
                incorrect_preds = predicted[incorrect_mask]
                
                for img, true_label, pred_label in zip(incorrect_images, incorrect_labels, incorrect_preds):
                    if len(images_to_show) >= num_images:
                        break
                    images_to_show.append(img.cpu())
                    labels_to_show.append(true_label.cpu().item())
                    predictions_to_show.append(pred_label.cpu().item())
                
                if len(images_to_show) >= num_images:
                    break
        
        title_prefix = "Misclassified Examples"
    else:
        # Show random examples
        images_to_show = []
        labels_to_show = []
        predictions_to_show = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                for i in range(min(num_images, len(images))):
                    images_to_show.append(images[i].cpu())
                    labels_to_show.append(labels[i].cpu().item())
                    predictions_to_show.append(predicted[i].cpu().item())
                break
        
        title_prefix = "Sample Predictions"
    
    # Plot the images
    plt.figure(figsize=(15, 6))
    for i in range(len(images_to_show)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images_to_show[i].squeeze(), cmap='gray')
        
        true_label = CLASS_LABELS[labels_to_show[i]]
        pred_label = CLASS_LABELS[predictions_to_show[i]]
        
        if labels_to_show[i] == predictions_to_show[i]:
            color = 'green'
            title = f'✓ {pred_label}'
        else:
            color = 'red'
            title = f'✗ True: {true_label}\nPred: {pred_label}'
        
        plt.title(title, color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f'{title_prefix}', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_model_comparisons(model_metrics):
    """Plot comparison of different models' performance"""
    models = list(model_metrics.keys())
    accuracies = [model_metrics[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies, color=['blue', 'red', 'green', 'orange', 'purple'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Plot F1 scores if available
    plt.subplot(1, 2, 2)
    if 'f1_macro' in model_metrics[models[0]]:
        f1_scores = [model_metrics[model]['f1_macro'] for model in models]
        bars = plt.bar(models, f1_scores, color=['blue', 'red', 'green', 'orange', 'purple'])
        plt.title('Model F1-Score Comparison')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def evaluate():
    """Evaluate all trained models"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on: {device}")
    
    # Get test data loader
    _, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS, apply_augmentation=False)
    
    # Define models to evaluate
    models_to_evaluate = [
        (MAIN_MODEL_NAME, FashionCNN, "Main CNN (Aug + Dropout)"),
        (BASELINE_MODEL_NAME, BaselineModel, "Baseline (Logistic Regression)"),
        (ALTERNATIVE_MODEL_NAME, AlternativeCNN, "Alternative CNN"),
        (NO_AUG_MODEL_NAME, FashionCNN, "CNN (No Augmentation)"),
        (NO_DROPOUT_MODEL_NAME, FashionCNN, "CNN (No Dropout)")
    ]
    
    model_metrics = {}
    
    for model_file, model_class, model_name in models_to_evaluate:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        model = load_model(model_file, model_class, device)
        if model is None:
            print(f"Skipping {model_name} due to loading error")
            continue
        
        metrics = evaluate_model(model, test_loader, device, plot_visuals=True)
        model_metrics[model_name] = metrics
        
        # Show sample predictions and errors
        print("\nSample Predictions:")
        visualize_model_predictions(model, test_loader, device, num_images=10, errors_only=False)
        
        print("\nMisclassified Examples:")
        visualize_model_predictions(model, test_loader, device, num_images=10, errors_only=True)
    
    # Plot model comparisons
    if model_metrics:
        print(f"\n{'='*50}")
        print("Model Comparison Summary")
        print(f"{'='*50}")
        plot_model_comparisons(model_metrics)
        
        # Print summary table
        print("\nSummary Table:")
        print("Model\t\t\t\tAccuracy\tF1-Score")
        print("-" * 55)
        for model_name, metrics in model_metrics.items():
            f1_score = metrics.get('f1_macro', 0.0)
            print(f"{model_name:<30}\t{metrics['accuracy']:.2f}%\t\t{f1_score:.4f}")

def evaluate_model(model, test_loader, device, plot_visuals=True):
    """
    Evaluate a single model and return comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Data loader for test dataset
        device: Device to run evaluation on
        plot_visuals: Whether to plot confusion matrix and ROC curves
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class metrics
    precision, recall, f1, support = calculate_per_class_metrics(all_labels, all_predictions)
    
    # Calculate macro averages
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    print(f"\nMacro Averages:")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-Score: {f1_macro:.4f}")
    
    if plot_visuals:
        # Plot confusion matrix
        plot_confusion_matrix(model, test_loader, device)
        
        # Plot ROC curves
        plot_roc_curves(all_labels, np.array(all_probabilities))
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support
    }

if __name__ == "__main__":
    evaluate()