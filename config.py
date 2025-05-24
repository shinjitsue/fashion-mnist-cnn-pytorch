"""
Configuration parameters for the Fashion MNIST CNN project.
"""

# Data parameters
DATA_DIR = './data'
BATCH_SIZE = 64  # Smaller batch size for CPU training
NUM_WORKERS = 0  # No extra workers for CPU
IMAGE_SIZE = 28  # Original Fashion MNIST image size

# Model parameters
NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
SAVE_DIR = './models'
MODEL_NAME = 'fashion_mnist_cnn.pth'

# Fashion MNIST class labels
CLASS_LABELS = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]