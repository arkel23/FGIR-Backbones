import math
import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange

from .matrix_sqrt import matrix_sqrt
from .mpncov import MPNCOV


class Head(nn.Module):
    def __init__(self, classifier, hidden_size, num_classes, bsd=True, proj_size=256):
        super().__init__()

        if classifier == 'mpncov':
            # class proj size for vgg/resnet by def is 256 (512 for vgg / 2048 for rn -> 256)
            # it uses a classifier factor (5, 100, 1000) that increases lr by factor for classifier
            self.mpncov = MPNCOV(input_dim=hidden_size, dimension_reduction=proj_size)
            self.head = nn.Sequential(
                Rearrange('b d 1 -> b d'),
                nn.Linear(proj_size * (proj_size + 1) // 2, num_classes)
            )
        elif classifier == 'iblp':
            # https://arxiv.org/abs/1707.06772
            # https://github.com/DennisLeoUTS/improved-bilinear-pooling/
            # the improved norm can be sqrt matrix or log matrix (not element-wise)
            self.matrix_sqrt = matrix_sqrt.apply
            self.blp_head = nn.Linear(hidden_size ** 2, num_classes)
        elif classifier == 'blp':
            # https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py
            self.blp_head = nn.Linear(hidden_size ** 2, num_classes)
        elif classifier == 'cls':
            self.head = nn.Linear(hidden_size, num_classes)
        else:
            self.head_pool = Reduce('b s d -> b d', 'mean')
            self.head = nn.Linear(hidden_size, num_classes)

        if not bsd:
            self.rearrange = Rearrange('b d h w -> b (h w) d')

    '''
        # Initialize weights
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        if hasattr(self, 'head'):
            nn.init.constant_(self.head.weight, 0)
            nn.init.constant_(self.head.bias, 0)
    '''

    def forward(self, x):
        # x shape: B (batch size), S (sequence length), D (hidden dim size) or B, D, H, W
        if hasattr(self, 'rearrange'):
            x = self.rearrange(x)

        if hasattr(self, 'mpncov'):
            # 2d input: b, c, h, w -> 1d output: b, dim_out*(dim_out+1)/2, 1
            x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.shape[1])))
            x = self.mpncov(x)
            x = self.head(x)

        elif hasattr(self, 'blp_head'):
            x = torch.matmul(rearrange(x, 'b s d -> b d s'), x) / x.shape[1]

            if hasattr(self, 'matrix_sqrt'):
                # matrix square root
                x = self.matrix_sqrt(x)

            x = rearrange(x, 'b d1 d2 -> b (d1 d2)')
            # https://github.com/pascal-niklaus/pascal/blob/master/pascal/R/sgnsqrt.R
            x = torch.sign(x) * torch.sqrt(torch.abs(x)) # + 1e-5
            x = torch.nn.functional.normalize(x)

            x = self.blp_head(x)

        elif hasattr(self, 'head_pool'):
            x = self.head_pool(x)
            x = self.head(x)

        elif hasattr(self, 'head'):
            x = self.head(x[:, 0, :])

        return x
