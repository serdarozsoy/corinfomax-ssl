import torch.nn as nn
import torchvision.models as models

class LinClassifier(nn.Module):
    """
    Linear Evaluation with a linear classifier defined as 1-layer:
        input size: feature dimension (Ex: 512 for ResNet18 )
        output size: number of class (Ex: 10 for CIFAR-10)
    Args:
        model_name (string): Backbone model name. Default as 'resnet-18'.
        dataset (string): Dataset used for linear evaluation. Default as 'cifar-10'
        x (torch.tensor): input with size (batchsize)x(features_dim)
    Returns:
        x (torch.tensor): logits with size (batchsize)x(num_classes). 
    """
    def __init__(self, args, offline=False):
        super().__init__()
        self.features_dim = getattr(models, args.model_name)().inplanes   # "inplanes" provides input size of last block for related resnet model. (Ex: 512 for ResNet18)
        if args.dataset == "cifar10":
            self.num_classes = 10
        elif args.dataset == "cifar100":
            self.num_classes = 100
        elif args.dataset == "tiny_imagenet":
            self.num_classes = 200
        elif args.dataset == "imagenet":
            self.num_classes = 1000
        elif args.dataset == "imagenet-100":
            self.num_classes = 100
        else:
            raise ValueError("given dataset name has not included yet!")
        self.classifier = nn.Linear(self.features_dim, self.num_classes)
        self.offline = offline  

    def forward(self, x):
        x = self.classifier(x)
        return x