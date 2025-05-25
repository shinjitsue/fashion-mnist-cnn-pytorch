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

# Model filenames
MAIN_MODEL_NAME = 'main_aug_dropout.pth'
BASELINE_MODEL_NAME = 'baseline_logistic.pth'
ALTERNATIVE_MODEL_NAME = 'alternative_shallow.pth'
NO_AUG_MODEL_NAME = 'main_no_aug.pth'
NO_DROPOUT_MODEL_NAME = 'main_no_dropout.pth'
BEST_MODEL_NAME = 'best_model.pth'

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