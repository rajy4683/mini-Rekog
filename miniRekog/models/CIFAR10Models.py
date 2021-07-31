
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


class CIFARModelDDilate(nn.Module):
    def __init__(self,dropout=0.1):
        super(CIFARModelDDilate, self).__init__()
        self.layer1_channels = 12
        self.dropout_val = dropout
        self.bias = False

        self.conv1 = nn.Sequential(
            # RF = 3
            nn.Conv2d(3, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), 
            #nn.Conv2d(3,self.layer1_channels,1,1,0,1,1,bias=bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            # RF = 5
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # RF = 9
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias),#,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),      
        )

        self.conv2 = nn.Sequential(
            # RF = 13
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), #groups=self.layer1_channels),
            #nn.Conv2d(self.layer1_channels,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            # RF = 17
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels*2),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            # RF = 25
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=2, stride=2, dilation=2, bias=self.bias,groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),      
        )

        self.conv3 = nn.Sequential(
            # RF=41
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias,dilation=2), #groups=self.layer1_channels*2),
            #nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),       
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # RF=57
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias, groups=self.layer1_channels*4, dilation=2),
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
        )

        self.conv4 = nn.Sequential(
            # RF=65
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),#, groups=self.layer1_channels),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # RF=73
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels),
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),

            # nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            nn.Conv2d(self.layer1_channels*8, 10,1,1,0,1,1,bias=self.bias),
        )
        
        self.gap_linear = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.AvgPool2d(kernel_size=8),
            # nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)

        return x

class STN_Cifar(nn.Module):
    def __init__(self,n_channels=3):
        super(STN_Cifar, self).__init__()
        # self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        # self.conv3_drop = nn.Dropout2d()
        self.custom_conv = CIFARModelDDilate()
        # self.fc1 = nn.Linear(500, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=3),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.Conv2d(8, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # # x = F.relu(self.conv3_drop(self.conv3(x)), 2)
        # x = x.view(-1, 500)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        return self.custom_conv(x)
        # return F.log_softmax(x, dim=1)


# model = Net(n_channels=3).to(device)