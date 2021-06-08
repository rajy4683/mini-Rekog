"""
     Models for MNIST digit recognition
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class MNISTLarge(nn.Module):
    """
        MNIST Large has ~9032 Parameters
    """
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_val = config.dropout
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),            
            nn.Dropout(self.dropout_val)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1,stride=1, bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val)
        )
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(16, 10, 1, bias=self.bias)
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

class MNISTMedium(nn.Module):
    """
        MNIST Medium has ~7288 Parameters
    """
    def __init__(self, dropout_val=0.1):
        super(Net, self).__init__()
        self.dropout_val = dropout_val
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias), # Input=8x28x28 Output=8x28x28 RF=5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # nn.Conv2d(8, 8, 3, padding=1, bias=self.bias),
            # nn.ReLU(),
            # nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),            # Input=8x28x28 Output=8x14x14 RF=6
            nn.Dropout(self.dropout_val),
            # nn.Conv2d(8, 8, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias), # Input=8x14x14 Output=8x14x14 RF=10
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            # nn.Conv2d(8, 16, 1),
            nn.Conv2d(8, 16, 3, padding=1, bias=self.bias), # Input=8x14x14 Output=16x14x14 RF=14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), # Input=16x14x14 Output=16x7x7 RF=16
            nn.Dropout(self.dropout_val),
            # nn.Conv2d(16, 16, 1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3,bias=self.bias), # Input=16x7x7 Output=16x5x5 RF=24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3,bias=self.bias), # Input=16x5x5 Output=16x3x3 RF=32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2), # Input=16x3x3 Output=16x1x1 RF=36
            nn.Dropout(self.dropout_val)
        )
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Conv2d(16, 10, 1, bias=self.bias)
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x


class MNISTSmall(nn.Module):
    """
        MNIST Small has ~5616 parameters
    """
    def __init__(self, dropout_val=0.1):
        super(MNISTSmall, self).__init__()
        self.dropout_val = dropout_val
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias), # Input=8x28x28 Output=8x28x28 RF=5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),            # Input=8x28x28 Output=8x14x14 RF=6
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias), # Input=8x14x14 Output=8x14x14 RF=10
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 16, 3, padding=1,stride=1, bias=self.bias), # Input=8x14x14 Output=8x14x14 RF=10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3, padding=1, stride=2,bias=self.bias), # Input=8x14x14 Output=16x14x14 RF=14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),
        )
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), # Input=16x1x1 Output=16x1x1 RF=36
            nn.Conv2d(16, 10, 1, bias=self.bias) # Input=16x1x1 Output=10x1x1 RF=36
        )                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x


class MNISTUltraSmall(nn.Module):
    """
        MNIST Small has ~4464 parameters
    """
    def __init__(self, dropout_val=0.1):
        super(MNISTUltraSmall, self).__init__()
        self.dropout_val = dropout_val
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias), # Input=1x28x28 Output=8x28x28 RF=3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=2,bias=self.bias), # Input=8x28x28 Output=8x28x28 RF=5
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),            # Input=8x28x28 Output=8x14x14 RF=6
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias), # Input=8x14x14 Output=8x14x14 RF=10
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias), # Input=8x14x14 Output=8x14x14 RF=10
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 16, 3, padding=1, stride=2,bias=self.bias), # Input=8x14x14 Output=16x14x14 RF=14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(self.dropout_val),

        )
               
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), # Input=16x1x1 Output=16x1x1 RF=36
            nn.Conv2d(16, 10, 1, bias=self.bias) # Input=16x1x1 Output=10x1x1 RF=36
        )
        self.final_linear = nn.Linear(10,10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)       
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = self.final_linear(x)
        x = F.log_softmax(x, dim=1)
        return x

