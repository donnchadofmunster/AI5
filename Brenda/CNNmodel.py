import torch
import torch.nn as nn
import torch.nn.functional as F

class LineFollower(nn.Module):
    def __init__(self, num_classes):  
        super(LineFollower, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=50, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=30, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(30 * 5 * 32, 128)  # Assuming input size is (50, 320)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x
    
