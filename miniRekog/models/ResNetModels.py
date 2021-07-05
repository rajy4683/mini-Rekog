import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicLNBlock(nn.Module):
    """
        Basic residual block class without bottleneck layers.
        This block forms the basic blocks for Resnet18 and Resnet34
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicLNBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(planes, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
                nn.GroupNorm(self.expansion*planes, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    """
        Basic residual block class without bottleneck layers.
        This block forms the basic blocks for Resnet18 and Resnet34
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
        Final Resnet Model generator.
        This class creates the final model architecture based on Block type and num_blocks array.
        By default, final classifier layer has classes count = CiFAR10 i.e 10 classes.
        For custom classes this parameter needs to be passed.
    """
    def __init__(self, block, num_blocks, num_classes=10, norm_type="BN"):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm_type=norm_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if self.norm_type == "LN":
            # self.bn1 = nn.BatchNorm2d(64)
            self.bn1 = nn.GroupNorm(64, 64)
        else :
            self.bn1 = nn.BatchNorm2d(64)

        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    """
        Returns the Resnet18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])
    # return ResNet(BasicLNBlock, [2, 2, 2, 2])

def ResNet34():
    """
        Returns the Resnet34 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNetLN18():
    """
        Returns the Resnet18 model
    """
    return ResNet(BasicLNBlock, [2, 2, 2, 2], norm_type="LN")

def ResNetLN34():
    """
        Returns the Resnet34 model
    """
    return ResNet(BasicLNBlock, [2, 2, 2, 2], norm_type="LN")



class ModifiedResBlock(nn.Module):
    """
        ModifiedResBlock: Class for creating Modified ResNet block. Based on S11:
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) 
        Add(X, R1)

    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ModifiedResBlock, self).__init__()
        self.layerconv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        ### This layer applies after the first conv and we intend to keep the channel size same
        self.resconv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        #self.shortcut = nn.Sequential() 

    def forward(self, x):
        out = self.layerconv(x)
        res = self.resconv(out)
        #out = res
        #out = F.relu(out)
        return out+res



class S8ResNet(nn.Module):
    """
        S8: Custom resnet block based model
        It used the ModifiedResBlock which doesnt have multiple layers.
        PrepLayer:
            Conv 3x3 s1, p1) >> BN >> RELU [64]
        Layer1:
            ModifiedResBlock(128)
        Layer 2:
            Conv 3x3 [256]
            MaxPooling2D
            BN
            ReLU
        Layer 3:
            ModifiedResBlock(512)
        MaxPooling:(with Kernel Size 4) 
        FC Layer 
        SoftMax
    """
    def __init__(self, num_classes=10,dropout=0.0):
        super(S8ResNet, self).__init__()
        self.in_planes = 64
        self.resize_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.prep_layer = nn.Sequential(
            nn.Conv2d(64, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU()
        )
        self.layer1 = ModifiedResBlock(self.in_planes, self.in_planes*2, 1)
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.in_planes*2, self.in_planes*4, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.in_planes*4),
            nn.ReLU()
        )
        self.layer3 = ModifiedResBlock(self.in_planes*4, self.in_planes*8, 1)
        self.layer4_supermax = nn.MaxPool2d(4,4)
        self.fc_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.resize_layer(x)
        out = self.prep_layer(out)
        #out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4_supermax(out)
        out = out.view(-1, 512)
        #out = out.view(-1, 10)
        out = self.fc_layer(out)
        #
        out = F.log_softmax(out, dim=1)
        return out
