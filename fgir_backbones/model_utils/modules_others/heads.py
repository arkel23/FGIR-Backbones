import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange


import torch
import torch.nn as nn
from torch.autograd import Function


def sqrt_newton_schulz(A, numIters):
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(torch.cuda.FloatTensor)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(torch.cuda.FloatTensor)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def lyap_newton_schulz(z, dldz, numIters):
    batchSize = z.shape[0]
    dim = z.shape[1]
    normz = z.mul(z).sum(dim=1).sum(dim=1).sqrt()
    a = z.div(normz.view(batchSize, 1, 1).expand_as(z))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(torch.cuda.FloatTensor)
    q = dldz.div(normz.view(batchSize, 1, 1).expand_as(z))
    for i in range(numIters):
        q = 0.5*(q.bmm(3.0*I - a.bmm(a)) - a.transpose(1, 2).bmm(a.transpose(1,2).bmm(q) - q.bmm(a)) )
        a = 0.5*a.bmm(3.0*I - a.bmm(a))
    dlda = 0.5*q
    return dlda


class matrix_sqrt(Function):
    @staticmethod
    def forward(ctx, input):
        output = sqrt_newton_schulz(input, 10)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = lyap_newton_schulz(output, grad_output, 10)
        return grad_input


class Head(nn.Module):
    def __init__(self, classifier, hidden_size, num_classes, bsd=True):
        super().__init__()

        if classifier == 'iblp':
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

        if hasattr(self, 'blp_head'):
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
