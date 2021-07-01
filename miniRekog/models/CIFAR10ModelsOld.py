
"""
     All model class definitions for CIFAR10
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class CIFARModelDoubleDilate(nn.Module):
    def __init__(self,dropout=0.1, layer1_channels =10):
        super(CIFARModelDoubleDilate, self).__init__()
        self.layer1_channels = 10
        self.dropout_val = dropout
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),
            #nn.Conv2d(3,self.layer1_channels,1,1,0,1,1,bias=bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),      
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            # # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), #groups=self.layer1_channels),
            #nn.Conv2d(self.layer1_channels,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels*2),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),      
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias,dilation=2), #groups=self.layer1_channels*2),
            #nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),       
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias, groups=self.layer1_channels*4, dilation=2),
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, stride=2,bias=self.bias, groups=self.layer1_channels*4, dilation=1),
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),#, groups=self.layer1_channels),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            nn.Conv2d(self.layer1_channels*8, 10,1,1,0,1,1,bias=self.bias),
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            # nn.Dropout(self.dropout_val)
            )
        
        self.gap_linear = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.AvgPool2d(kernel_size=8),
            # nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)

        return x


class CIFARModelDDilate(nn.Module):
    def __init__(self,dropout=0.1):
        super(CIFARModelDDilate, self).__init__()
        self.layer1_channels = 12
        self.dropout_val = dropout
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),
            #nn.Conv2d(3,self.layer1_channels,1,1,0,1,1,bias=bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),      
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            # # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(self.dropout_val)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), #groups=self.layer1_channels),
            #nn.Conv2d(self.layer1_channels,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels*2),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),      
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(self.dropout_val)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias,dilation=2), #groups=self.layer1_channels*2),
            #nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),       
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias, groups=self.layer1_channels*4, dilation=2),
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, stride=2,bias=self.bias, groups=self.layer1_channels*4, dilation=1),
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(self.dropout_val)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),#, groups=self.layer1_channels),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            nn.Conv2d(self.layer1_channels*8, 10,1,1,0,1,1,bias=self.bias),
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            # nn.Dropout(self.dropout_val)
        )
        
        self.gap_linear = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.AvgPool2d(kernel_size=8),
            # nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)

        return x