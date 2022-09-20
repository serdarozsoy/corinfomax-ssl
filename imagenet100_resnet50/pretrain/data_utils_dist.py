import torch
import torchvision
import transform_utils3 as utils
import random


def make_data(dataset, subset, subset_type):
    # Data loading and preperation
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
    if dataset == 'imagenet':
        pretrain_data = torchvision.datasets.ImageFolder(root="~/data/train",  \
                                                transform=utils.PretrainTransform(dataset))
        lin_train_data = torchvision.datasets.ImageFolder(root="~/data/train",  \
                                                transform=utils.EvalTransform(dataset,train_transform = True))
        lin_test_data = torchvision.datasets.ImageFolder(root="~/data/val", \
                                                transform=utils.EvalTransform(dataset,train_transform = False))    
    if dataset == 'imagenet-100':
        pretrain_data = torchvision.datasets.ImageFolder(root="~/data/imagenet-100/train",  \
                                                transform=utils.PretrainTransform(dataset))
        lin_train_data = torchvision.datasets.ImageFolder(root="~/data/imagenet-100/train",  \
                                                transform=utils.EvalTransform(dataset,train_transform = True))
        lin_test_data = torchvision.datasets.ImageFolder(root="~/data/imagenet-100/val", \
                                                transform=utils.EvalTransform(dataset,train_transform = False))       
    
    if (subset_type == "linear") & (subset<1.0):
        random.seed(4) # Work with same subset for each experiment
        t_classes = lin_train_data.classes
        subset_list = []
        cls_no = len(lin_train_data.classes)
        for k in range(cls_no):
            new_list = random.sample([i for i, j in enumerate(lin_train_data.targets) if j == k], int(len(lin_train_data)/cls_no*subset))
            subset_list = subset_list + new_list
        lin_train_data  = torch.utils.data.Subset(lin_train_data, subset_list)
        lin_train_data.classes = t_classes

    if (subset_type == "pretrain_and_linear") & (subset<1.0):
        random.seed(4)
        t_classes = pretrain_data.classes
        subset_list = []
        cls_no = len(pretrain_data.classes)
        for k in range(cls_no):
            new_list = random.sample([i for i, j in enumerate(pretrain_data.targets) if j == k], int(len(pretrain_data)/cls_no*subset))
            subset_list = subset_list + new_list
        pretrain_data  = torch.utils.data.Subset(pretrain_data, subset_list)
        pretrain_data.classes = t_classes
        
        lin_train_data  = torch.utils.data.Subset(lin_train_data, subset_list)
        lin_train_data.classes = t_classes                     
                                           

    return  pretrain_data, lin_train_data, lin_test_data