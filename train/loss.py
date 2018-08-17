import torch.nn as nn
import torch.nn.functional as F


class BCEWeightLoss(nn.Module):

    def __init__(self):
        super(BCEWeightLoss, self).__init__()

    def forward(self, input, target, weight=None):
        return F.binary_cross_entropy_with_logits(input, target, weight, size_average=False)