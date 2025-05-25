import os
import time
from train import train, train_with_different_lr
from evaluate import evaluate
from utils import plot_sample_images, get_data_loaders
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, SAVE_DIR

def main():
    """Main function to run the whole pipeline"""
    start_time = time.time()
    print("Fashion-MNIST CNN Classification")
    print("=" * 30)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Get data loaders
    train_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    # Visualize sample images from dataset
    print("\nVisualizing sample images from dataset...")
    plot_sample_images(train_loader)
    
    print("\nTraining models:")
    
    # Train main model (with augmentation and dropout)
    print("\n1. Training main CNN with augmentation and dropout")
    train(model_type="cnn", use_augmentation=True, use_dropout=True)
    
    # Train baseline model (logistic regression)
    print("\n2. Training baseline model")
    train(model_type="baseline", use_augmentation=False, use_dropout=False)
    
    # Train alternative CNN
    print("\n3. Training alternative CNN")
    train(model_type="alternative", use_augmentation=True, use_dropout=True)
    
    # Ablation study: without augmentation
    print("\n4. Ablation study: Training without data augmentation")
    train(model_type="cnn", use_augmentation=False, use_dropout=True)
    
    # Ablation study: without dropout
    print("\n5. Ablation study: Training without dropout")
    train(model_type="cnn", use_augmentation=True, use_dropout=False)
    
    # Ablation study: different learning rates
    print("\n6. Ablation study: Learning Rate Comparison")
    train_with_different_lr(learning_rates=[0.01, 0.001, 0.0001])
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluate()
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = ["torch", "torchvision", "matplotlib", "numpy", "sklearn", "seaborn", "tqdm"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Warning: Missing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        return False
    return True

if __name__ == "__main__":
    if check_dependencies():
        main()
    else:
        print("Please install missing dependencies before running.")
