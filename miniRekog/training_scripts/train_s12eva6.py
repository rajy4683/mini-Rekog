import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('/content/mini-Rekog/')
from miniRekog.config.config import config
from miniRekog.models.CIFAR10Models import STN_Cifar
from miniRekog.utils import fileutils
from miniRekog.dataloaders import dataloader
from miniRekog.train.traintest2 import execute_model

from torchsummary import summary
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
import torchvision.transforms as transforms
import torchvision
from torch.optim.lr_scheduler import (StepLR,
    OneCycleLR, MultiStepLR, CyclicLR, ReduceLROnPlateau
)
from albumentations import (
    HorizontalFlip, Compose, RandomCrop, Cutout,Normalize,ShiftScaleRotate, CoarseDropout, 
    Resize,RandomSizedCrop, MotionBlur,PadIfNeeded,Flip, IAAFliplr,ToGray,Rotate,
)

from albumentations.pytorch import ToTensorV2

saved_model_path=None
def main():
    global config
    for k,v in config.get_dict().items():
        print(f"{k}: {v}")
    device = torch.device("cuda" if not config['no_cuda'] else "cpu")
    
    print("Initializing datasets and dataloaders")

    torch.manual_seed(config['seed'])
    transform_train = Compose([
        # PadIfNeeded(min_height=40, min_width=40,p=1,always_apply=True),
        # RandomCrop(32,32,p=1,always_apply=True),
        # HorizontalFlip(p=0.5),
        # # PadIfNeeded(min_height=32, min_width=32,p=1,always_apply=True),
        # Cutout(1, 8,8,
        #     fill_value=[0.49139968*255, 0.48215841*255, 0.44653091*255],
        #     always_apply=False, p=1),
        
        Normalize(
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
        ),
        # ToGray(p=1),
        ToTensorV2()
    ])

    transform_test = Compose([
        Normalize(
        mean=[0.49421428, 0.48513139, 0.45040909],
        std=[0.24665252, 0.24289226, 0.26159238],
        ),
        ToTensorV2()
    ])
    trainloader, testloader = dataloader.get_train_test_dataloader_cifar10(transform_train=transform_train, 
                                                                        transform_test=transform_test,
                                                                        config=config)
    model_new = STN_Cifar(n_channels=3)
   
    optimizer=optim.SGD(model_new.parameters(), 
                        lr=config.lr,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 
    #                               mode='min', 
    #                               factor=0.2, 
    #                               patience=2, 
    #                               verbose=True, 
    #                               threshold=0.0001)

    criterion=nn.CrossEntropyLoss
    cycle_momentum = True if config.cycle_momentum == "True" else False
    print("Momentum cycling set to {}".format(cycle_momentum))
    scheduler = OneCycleLR(optimizer, 
                            config.ocp_max_lr, 
                            epochs=config.epochs, 
                            cycle_momentum=cycle_momentum, 
                            steps_per_epoch=len(trainloader), 
                            base_momentum=config.momentum,
                            max_momentum=0.95, 
                            pct_start=0.208,
                            anneal_strategy=config.anneal_strategy,
                            div_factor=config.div_factor,
                            final_div_factor=config.final_div_factor
                           )
    

    final_model_path = execute_model(model_new, 
                config, 
                trainloader, 
                testloader, 
                device, 
                dataloader.cifar10_classes,
                optimizer_in=optimizer,
                scheduler=scheduler,
                prev_saved_model=saved_model_path,
                criterion=criterion,
                save_best=True,
                batch_step=True)


if __name__ == '__main__':
    global config
    parser = argparse.ArgumentParser(description = 'Train CIFAR')
    #parser.add_argument("-h", "--help", required=False, help="Can be used to manipulate load-balancing")
    parser.add_argument("-p", "--params", required=False, help="JSON format string of params E.g: '{\"lr\":0.01, \"momentum\": 0.9}' ")
    parser.add_argument("-r", "--saved_model_path", required=False, help="Load and resume model from this path ")
    
    args = parser.parse_args()
    
   
    if (args.saved_model_path is not None):
        saved_model_path = args.saved_model_path
        print("Model will be loaded from",saved_model_path)
    if (args.params is not None):
        #if(args.params == "params"):    
        arg_val_dict = json.loads(args.params)
        #print(config['lr'])
        for key,val in arg_val_dict.items():
            print("Setting ",key," = ",val)
            config.set(key,val)
        print("Final Hyperparameters")

        main()
    #return