import torch
import torchvision
import torchvision.transforms as transforms

from albumentations import (
    HorizontalFlip, Compose, RandomCrop, Cutout,Normalize, 
    HorizontalFlip, RandomBrightnessContrast, CoarseDropout, ToGray,
    Resize,RandomSizedCrop, MotionBlur,InvertImg, IAAFliplr, ShiftScaleRotate,
	IAAPerspective,
)
from albumentations.pytorch import ToTensorV2
import random

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import sys
import numpy as np
import pandas as pd
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from . import imgnetloader

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


cifar10_classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MyCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(MyCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


## Dummy wrapper around 
def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return torch.utils.data.DataLoader(dataset, batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

def get_default_transforms_cifar10(seed=0xdeadbeef):
    torch.manual_seed(seed)
    patch_size=2
    transform_train = Compose([
    Cutout(num_holes=1,max_h_size=16,max_w_size=16,always_apply=True,p=1,fill_value=[0.4819*255, 0.4713*255, 0.4409*255]),
    IAAFliplr(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True, p=1),
    Normalize(
       mean=[0.4914, 0.4826, 0.44653],
       std=[0.24703, 0.24349, 0.26519],
       ),
    # Normalize(
    #    mean=[0.5268, 0.5267, 0.5328],
    #    std=[0.3485, 0.3444, 0.3447],
    #    ),
    
    ToTensorV2()
    ])

    transform_test = Compose([
    Normalize(
        mean=[0.4914, 0.4826, 0.44653],
        std=[0.24703, 0.24349, 0.26519],
    ),
    ToTensorV2()
    ])
    return transform_train, transform_test


def get_updated_default_transforms_cifar10(seed=0xdeadbeef):
    torch.manual_seed(seed)
    transform_train = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        CoarseDropout(max_holes=1, 
                    max_height=16,
                    max_width=16, 
                    min_holes=1, 
                    min_height=16, 
                    min_width=16, 
                    fill_value=[0.49139968, 0.48215841, 0.44653091], 
                    mask_fill_value=None, 
                    always_apply=False, p=0.5),
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

    return transform_train, transform_test


def get_normalizer_transform(dataset_type,transforms_type):
    if dataset_type == "imagenet":
        return transforms_type.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_type == "cifar10":
        return transforms_type.Normalize(mean=[0.4914, 0.4826, 0.44653], std=[0.24703, 0.24349, 0.26519])

def get_minimal_transforms(dataset_type, transforms_type, seed=0xdeadbeef):
    torch.manual_seed(seed)
    norm_transform = get_normalizer_transform(dataset_type,transforms_type)
    transform_train = transforms_type.Compose([
        norm_transform,
        transforms_type.ToTensorV2()
    ])

    transform_test = transforms_type.Compose([
        norm_transform,
        transforms_type.ToTensorV2()
    ])
    return transform_train, transform_test


def get_train_test_dataloader_cifar10(transform_train=None, 
                                      transform_test=None,
                                      config=None,
                                      seed=0xdeadbeef):
    
    if config is None:
        raise Exception

    torch.manual_seed(seed)
    
    transform_train_def, transform_test_def = get_default_transforms_cifar10()
    if (transform_train is None):
        transform_train = transform_train_def
    if (transform_test is None):
        transform_test = transform_test_def
    
    trainset = MyCIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = get_dataloader(trainset, 
                                 config['batch_size'], 
                                 shuffle=True, 
                                 num_workers=2)
    testset = MyCIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = get_dataloader(testset, 
                                config['batch_size'], 
                                shuffle=False, 
                                num_workers=2)
    return trainloader, testloader    


def get_imagenet_loaders(train_path, 
                         test_path,
                         config=None,
                         transform_train=None, 
                         transform_test=None,
                         seed=0xdeadbeef):
    '''
    Returns custom imagenet loader given the train and test path where the imagenet data is stored.

    '''
    torch.manual_seed(seed)
    
    train_data, test_data = imgnetloader.generate_timgnet_train_test_data(train_path, 
                                                                          0.7,
                                                                          transform_train, 
                                                                          transform_test)
    print(train_data.transform, test_data.transform)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=config.get('batch_size'),
                                              shuffle=True,
                                              **kwargs)
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=config.get('batch_size'),
                                             shuffle=True,
                                             **kwargs)
    
    return trainloader, testloader
    