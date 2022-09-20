
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms 
import random
import numpy


"""
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)
"""

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)


class PretrainTransform:
    def __init__(self, dataset): 
        self.min_scale = 0.08 #0.2 #0.08 in some versions
        if dataset=="cifar10":
            self.data_normalize_mean = (0.4914, 0.4822, 0.4465)
            self.data_normalize_std = (0.247, 0.243, 0.261)
            self.random_crop_size = 32
            self.gaussian_prob = 0 

        elif dataset=="cifar100":
            self.data_normalize_mean = (0.5071, 0.4865, 0.4409)
            self.data_normalize_std = (0.2673, 0.2564, 0.2762)
            self.random_crop_size = 32
            self.gaussian_prob = 0
        elif dataset=="tiny_imagenet":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 64
            self.gaussian_prob = 0.5
        elif dataset=="imagenet":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 224
            self.gaussian_prob = 0.5
        elif dataset=="imagenet-100":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 224
            self.gaussian_prob = 0.5
        else:
            raise ValueError('Dataset is not normalized!')

        self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            self.random_crop_size, 
                            scale=(self.min_scale, 1.0),
                            interpolation=transforms.InterpolationMode.BICUBIC, # Only in VicReg
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur()], p=1.0), # only for TinyImageNet
                        transforms.RandomApply([Solarization()], p=0.0), # Only in VicReg
                        transforms.ToTensor(),
                        transforms.Normalize(self.data_normalize_mean, self.data_normalize_std),
                    ]
                )

        self.transform_prime = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            self.random_crop_size, 
                            scale=(self.min_scale, 1.0),
                            interpolation=transforms.InterpolationMode.BICUBIC, # Only in VicReg
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                        ),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([GaussianBlur()], p=0.1), # only for TinyImageNet
                        transforms.RandomApply([Solarization()], p=0.2), # Only in VicReg
                        transforms.ToTensor(),
                        transforms.Normalize(self.data_normalize_mean, self.data_normalize_std),
                    ]
                )



    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class EvalTransform:
    def __init__(self, dataset, train_transform = True):
        if dataset =="cifar10":
            self.data_normalize_mean = (0.4914, 0.4822, 0.4465)
            self.data_normalize_std = (0.247, 0.243, 0.261)
            self.random_crop_size = 32
        elif dataset =="cifar100":
            self.data_normalize_mean = (0.5071, 0.4865, 0.4409)
            self.data_normalize_std = (0.2673, 0.2564, 0.2762)
            self.random_crop_size = 32
        elif dataset=="tiny_imagenet":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 64
        elif dataset=="imagenet":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 224
        elif dataset=="imagenet-100":
            self.data_normalize_mean = (0.485, 0.456, 0.406)
            self.data_normalize_std = (0.229, 0.224, 0.225)
            self.random_crop_size = 224
        else:
            raise ValueError('Dataset is not normalized!')
            
        if train_transform is True:
            self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.random_crop_size, interpolation=transforms.InterpolationMode.BICUBIC), # scale=(0.2, 1.0) is possible
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.data_normalize_mean, self.data_normalize_std),
            ]
        )
        else:
            self.transform = transforms.Compose(
            [   
                transforms.Resize(int(self.random_crop_size*(8/7)), interpolation=transforms.InterpolationMode.BICUBIC), # In Imagenet: 224 -> 256 
                transforms.CenterCrop(self.random_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.data_normalize_mean, self.data_normalize_std),
            ]
        )

    def __call__(self, x):
        return self.transform(x)



