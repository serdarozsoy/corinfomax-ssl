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
        prefix = IMAGENET_DATA_LOCATION_PREFIX_MAP[dataset_location]
        if dataset_location=="shm":
            if not os.path.exists(prefix) or len(os.listdir(prefix)) == 0:
                print(f"Dataset not found in {prefix}. Please copy the dataset to {prefix} by using the following command:")
                default_path = os.path.join(IMAGENET_DATA_LOCATION_PREFIX_MAP['default'], "*")
                print(f"rsync -ahvc --progress {default_path} {prefix}")
                exit(1)
        pretrain_data = torchvision.datasets.ImageFolder(root= os.path.join(prefix, "train"),  \
                                                transform=utils.PretrainTransform(dataset))
        lin_train_data = torchvision.datasets.ImageFolder(root=os.path.join(prefix, "train"),  \
                                                transform=utils.EvalTransform(dataset,train_transform = True))
        lin_test_data = torchvision.datasets.ImageFolder(root=os.path.join(prefix, "val"), \
                                                transform=utils.EvalTransform(dataset,train_transform = False))        
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
        lin_train_data = torchvision.datasets.ImageFolder(root=os.path.join(prefix, "train"),  \
                                                transform=utils.EvalTransform(dataset,train_transform = True))
        lin_test_data = torchvision.datasets.ImageFolder(root=os.path.join(prefix, "val"), \
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