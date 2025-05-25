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
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Show sample images from dataset
    print("Plotting sample images...")
    train_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    plot_sample_images(train_loader)
    
    # 2. Train models
    print("\nTraining main CNN (with augmentation and dropout)...")
    train(model_type="cnn", use_augmentation=True, use_dropout=True)
    
    print("\nTraining baseline model...")
    train(model_type="baseline")
    
    print("\nTraining alternative CNN architecture...")
    train(model_type="alternative")
    
    print("\nTraining CNN without augmentation...")
    train(model_type="cnn", use_augmentation=False, use_dropout=True)
    
    print("\nTraining CNN without dropout...")
    train(model_type="cnn", use_augmentation=True, use_dropout=False)
    
    # 3. Learning rate study
    print("\nConducting learning rate study...")
    train_with_different_lr()
    
    # 4. Evaluate models
    print("\nEvaluating all models...")
    evaluate()
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


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
