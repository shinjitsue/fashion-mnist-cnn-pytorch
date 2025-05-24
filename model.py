"""
CNN model architecture for Fashion MNIST classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FashionCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion MNIST classification.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    """
    
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()
        
        # First convolutional layer
        # Input: 1 channel (grayscale), Output: 32 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 window
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        # Input: 32 channels, Output: 64 feature maps, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 window
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two 2x2 pooling layers, the 28x28 image becomes 7x7
        # Number of features = 64 feature maps * 7 * 7
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # First conv block: conv -> relu -> pooling
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Second conv block: conv -> relu -> pooling
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layer
        # -1 in the reshape corresponds to the batch size
        x = x.view(-1, 64*7*7)
        
        # First fully connected layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer
        x = self.fc2(x)
        
        return x
