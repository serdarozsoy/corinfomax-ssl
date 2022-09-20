import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import dbn 


class CovModel6(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        
        if (args.model_name == "resnet18") & (args.dataset in ["imagenet-100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("given setting has not included yet!")

        # projector
        if (args.model_name == "resnet18") & (args.dataset in ["imagenet-100"]):
            sizes = [512] + list(map(int, args.projector.split('-'))) # 512 = self.feature_dim
        else:
            raise ValueError("given setting has not included yet!")

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

        if self.normalized:
            z1 = F.normalize(z1, p=2)
            z2 = F.normalize(z2, p=2)

        return z1, z2



class LinModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if (args.model_name == "resnet18") & (args.dataset in ["imagenet-100"]):
            self.backbone = resnet18(zero_init_residual=True)
            self.backbone.fc = nn.Identity() 

    def forward(self, x):
        feature = self.backbone(x)
        #feature = torch.flatten(feature, start_dim=1)
        return feature