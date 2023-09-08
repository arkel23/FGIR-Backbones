# https://arxiv.org/abs/1707.06772
# https://github.com/DennisLeoUTS/improved-bilinear-pooling/
# https://github.com/HaoMood/blinear-cnn-faster/blob/master/src/model.py

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable


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
