import torch
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy

from .focal_loss import FocalLoss
from .mix import mixup_criterion


# Center Loss for Attention Regularization
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, output, targets):
        return self.l2_loss(output, targets) / output.size(0)


# Overall CAL Loss
class CALLoss(nn.Module):
    def __init__(self):
        super(CALLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()

    def forward(self, output, y):
        if len(output) == 7:
            (_, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
             y_pred_aug, _) = output

            y_aug = torch.cat([y, y], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)
 
            batch_loss = (self.cross_entropy_loss(y_pred_raw, y) / 3. +
                          self.cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. +
                          self.cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. +
                          self.center_loss(feature_matrix, feature_center_batch))

        elif isinstance(output, tuple) and len(output) == 2:
            y_pred, _ = output
            batch_loss = self.cross_entropy_loss(y_pred, y)

        else:
            batch_loss = self.cross_entropy_loss(output, y)

        return batch_loss


class OverallLoss(nn.Module):
    def __init__(self, args):
        super(OverallLoss, self).__init__()

        self.args = args

        if args.selector == 'cal':
            self.criterion = CALLoss()
        elif args.focal_gamma:
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

        if self.args.selector == 'cal':
            if len(output) == 7:
                output, _, _, _, _, _, _ = output
            elif isinstance(output, tuple) and len(output) == 2:
                output, _ = output

        return output, loss
