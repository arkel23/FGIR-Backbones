import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from timm.loss import LabelSmoothingCrossEntropy

from .focal_loss import FocalLoss
from .mix import mixup_criterion

class OverallLoss(nn.Module):
    def __init__(self, args):
        super(OverallLoss, self).__init__()

        self.args = args

        if args.focal_gamma:
            self.criterion = FocalLoss(args.focal_gamma, smoothing=args.ls)
        elif args.ls:
            self.criterion = LabelSmoothingCrossEntropy(args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, output, targets, y_a=None, y_b=None, lam=None):

        if y_a is not None:
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            loss = self.criterion(output, targets)

        return output, loss
