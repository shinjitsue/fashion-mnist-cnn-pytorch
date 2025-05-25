import os
from train import train
from evaluate import evaluate
from utils import plot_sample_images, get_data_loaders
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

def main():
    """Main function to run the whole pipeline"""
    print("Fashion-MNIST CNN Classification")
    print("=" * 30)
    
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
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluate()

if __name__ == "__main__":
    main()