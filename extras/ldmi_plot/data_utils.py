import torch
import torchvision
import transform_utils as utils
import random


def make_data(dataset, subset, subset_type):
    """Build data folder, includes defined image augmentations in utils as function.
    """
    if dataset == 'cifar10':
        pretrain_data = torchvision.datasets.CIFAR10(root="data", train=True, \
                                                transform=utils.PretrainTransform(dataset), download=True)
        lin_train_data = torchvision.datasets.CIFAR10(root="data", train=True, \
                                                transform=utils.EvalTransform(dataset,train_transform = True), download=True)

        lin_test_data = torchvision.datasets.CIFAR10(root="data", train=False, \
                                                transform=utils.EvalTransform(dataset,train_transform = False), download=True)
    if dataset == 'cifar100':
        pretrain_data = torchvision.datasets.CIFAR100(root="data", train=True, \
                                                transform=utils.PretrainTransform(dataset), download=True)
        lin_train_data = torchvision.datasets.CIFAR100(root="data", train=True, \
                                                transform=utils.EvalTransform(dataset,train_transform = True), download=True)
        lin_test_data = torchvision.datasets.CIFAR100(root="data", train=False, \
                                                transform=utils.EvalTransform(dataset,train_transform = False), download=True)
    if dataset == 'tiny_imagenet':
        pretrain_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train",  \
                                                transform=utils.PretrainTransform(dataset))
        lin_train_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train",  \
                                                transform=utils.EvalTransform(dataset,train_transform = True))
        lin_test_data = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/val", \
                                                transform=utils.EvalTransform(dataset,train_transform = False))                   
                                           
    return  pretrain_data, lin_train_data, lin_test_data 