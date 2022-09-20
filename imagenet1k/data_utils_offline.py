import torch
import torchvision
import transform_utils3 as utils
import random
import os

IMAGENET_DATA_LOCATION_PREFIX_MAP = {
    "default": "~/data/",
}


IMAGENET_100_DATA_LOCATION_PREFIX_MAP = {
    "default": "~/imagenet_100_data/",
}

def make_data(dataset, subset, subset_type, dataset_location="default"):
    # Data loading and preperation
    if dataset == 'cifar10':
        pretrain_data = torchvision.datasets.CIFAR10(root="data", train=True, \
                                                transform=utils.PretrainTransform(dataset), download=True)

    if dataset == 'cifar100':
        pretrain_data = torchvision.datasets.CIFAR100(root="data", train=True, \
                                                transform=utils.PretrainTransform(dataset), download=True)

    if dataset == 'tiny_imagenet':
        pretrain_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train",  \
                                                transform=utils.PretrainTransform(dataset))

    if dataset == 'imagenet':
        prefix = IMAGENET_DATA_LOCATION_PREFIX_MAP[dataset_location]
        if dataset_location=="shm":
            if not os.path.exists(prefix) or len(os.listdir(prefix)) == 0:
                print(f"Dataset not found in {prefix}. Please copy the dataset to {prefix} by using the following command:")
                default_path = os.path.join(IMAGENET_DATA_LOCATION_PREFIX_MAP['default'], "*")
                print(f"rsync -ahvc --progress {default_path} {prefix}")
                exit(1)
        pretrain_data = torchvision.datasets.ImageFolder(root= os.path.join(prefix, "train"),  \
                                                transform=utils.PretrainTransform(dataset))     
    if dataset == 'imagenet-100':
        prefix = IMAGENET_100_DATA_LOCATION_PREFIX_MAP[dataset_location]
        if dataset_location=="shm":
            if not os.path.exists(prefix) or len(os.listdir(prefix)) == 0:
                print(f"Dataset not found in {prefix}. Please copy the dataset to {prefix} by using the following command:")
                default_path = os.path.join(IMAGENET_100_DATA_LOCATION_PREFIX_MAP['default'], "*")
                print(f"rsync -ahvc --progress {default_path} {prefix}")
                exit(1)
        pretrain_data = torchvision.datasets.ImageFolder(root= os.path.join(prefix, "train"),  \
                                                transform=utils.PretrainTransform(dataset))
                   
                                           

    return  pretrain_data