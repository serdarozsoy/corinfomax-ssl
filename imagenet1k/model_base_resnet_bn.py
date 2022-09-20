import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
import dbn 

class LdmiModel(nn.Module):
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

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1])) #, bias=False
        layers.append(nn.Tanh())
        self.projector = nn.Sequential(*layers)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        return z1, z2

class CovModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity()
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1],bias=False)) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
           z = torch.cat((z1, z2), 0)
           #z = F.normalize(z, p=2, dim=0) # previous version
           z = F.normalize(z)
           z1, z2 = torch.split(z, z1.size()[0])

        return z1, z2


class LinModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity() 
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

    def forward(self, x):
        feature = self.backbone(x)
        #feature = torch.flatten(feature, start_dim=1)
        return feature

class LinModel5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity() 
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

    def forward(self, x):
        x = self.backbone(x)
        #feature = torch.flatten(feature, start_dim=1)
        return x




class ExpModel(nn.Module):
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

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1])) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
           z = torch.cat((z1, z2), 0)
           z = F.normalize(z, p=2, dim=0)
           z1, z2 = torch.split(z, z1.size()[0])

        return z1, z2


class ExpModelBN(nn.Module):
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

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.BatchNorm1d(sizes[-1],affine=False)) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
           z = torch.cat((z1, z2), 0)
           z = F.normalize(z, p=2, dim=0)
           z1, z2 = torch.split(z, z1.size()[0])

        return z1, z2


class ExpModelDBN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity()
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(dbn.DBN(sizes[-1], num_groups=128, dim=2, eps=1e-10, momentum=0.9, affine=False)) #, num_groups=2, mode=1
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
           z = torch.cat((z1, z2), 0)
           z = F.normalize(z, p=2, dim=0)
           z1, z2 = torch.split(z, z1.size()[0])

        return z1, z2


class CovModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity()
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet","imagenet-100"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet","imagenet","imagenet-100"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1],bias=False)) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2


class CovModel3(nn.Module):
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

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        #for i in range(len(sizes) - 2):
        #    layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
        #    layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #    layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1])) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2

class CovModel6(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity() 
            self.backbone.fc = nn.Identity()
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet", "imagenet", "imagenet-100"]):
            self.backbone = resnet50(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["cifar10","cifar100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        if (args.model_name == "resnet50") & (args.dataset in ["tiny_imagenet", "imagenet", "imagenet-100"]):
            sizes = [2048] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) #, bias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1],bias=False)) #, bias=False
        self.projector = nn.Sequential(*layers)
        self.normalized = args.normalize_on
        print(self.normalized)


    def forward(self, y1, y2):
        feature_1 = self.backbone(y1)
        z1 = self.projector(feature_1)
        feature_2 = self.backbone(y2)
        z2 = self.projector(feature_2)

        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2
