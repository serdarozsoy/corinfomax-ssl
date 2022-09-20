import torch
import torch.nn as nn
from torch.nn import Parameter


class DBN(nn.Module):
    def __init__(self, num_features, num_groups=32, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        size = input.size()
        assert input.dim() == self.dim and size[1] == self.num_features
        x = input.view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input.device)
            # print('sigma size {}'.format(sigma.size()))
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)