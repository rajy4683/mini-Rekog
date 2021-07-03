# mini-Rekog
PyTorch based library for training Vision and Text Models. Below is the description of the repo:

```
├── README.md    ### main README
├── miniRekog
│   ├── config           ### configuration and hyperparameters
│   │   ├── __init__.py
│   │   └── config.py
│   ├── dataloaders      ### wrappers around various datasets
│   │   ├── __init__.py
│   │   └── dataloader.py 
│   ├── losses             ### custom loss functions - Currently NotImplemented
│   │   └── __init__.py 
│   ├── misc
│   │   └── __init__.py
│   ├── models         ### models directory
│   │   ├── CIFAR10Models.py    ### all CIFAR10 models
│   │   ├── CIFAR10ModelsOld.py   ### deprecated CIFAR10 Models
│   │   ├── MNISTModels.py   ### all MNIST Models
│   │   ├── ResNetModels.py  ### all ResNet based models
│   │   └── __init__.py
│   ├── train
│   │   ├── __init__.py
│   │   ├── traintest.py    ### training/test functions with supporting utilities
│   │   └── traintest2.py  ### updated training/test functions
│   ├── training_scripts
│   │   ├── __init__.py
│   │   └── train_s8eva6.py  ### single script to invoke ResNet based models on CIFAR10
│   └── utils          ### All Utilities
│       ├── __init__.py
│       ├── fileutils.py        ### basic utility functions used across this Repo`
│       ├── gradcam2.py    		### gradcam utilities`
│       └── logger.py       	### basic logger functions. Currently based on wandb.ai`
├── notebooks
│   ├── __init__.py
│   └── samples.ipynb 	### sample notebook for using this Repo`
└── requirements.txt  
```



