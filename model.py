import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion MNIST classification.
    
    Architecture:
    - 3 Convolutional layers with batch normalization, ReLU activation and max pooling
    - 2 Fully connected layers with dropout
    """
    
    def __init__(self, num_classes=10, use_dropout=True):
        super(FashionCNN, self).__init__()
        
        self.use_dropout = use_dropout
        dropout_rate = 0.25 if use_dropout else 0.0
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block (new)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # No pooling needed here to maintain feature map size
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128*7*7, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # First conv block: conv -> batch norm -> relu -> pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: conv -> batch norm -> relu -> pooling
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block: conv -> batch norm -> relu
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128*7*7)
        
        # First fully connected layer with dropout
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x

class BaselineModel(nn.Module):
    """Simple baseline model (logistic regression)"""
    def __init__(self, num_classes=10):
        super(BaselineModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

class AlternativeCNN(nn.Module):
    """Alternative shallow CNN architecture for comparison"""
    def __init__(self, num_classes=10):
        super(AlternativeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x