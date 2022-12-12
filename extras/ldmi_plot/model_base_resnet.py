import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
import math


class CovModel(nn.Module):
    """Encoder and projection networks which are build our self-supervised learning setup.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # encoder network setting         
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity()
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

        # projector network setting
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 2048 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) 
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1],bias=False))
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        # l-2 normalization of projector output 
        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2


class LinModel(nn.Module):
    """Base encoder network for loading pretrained model weights for linear evaluation
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity() 
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

    def forward(self, x):
        feature = self.backbone(x)
        return feature



