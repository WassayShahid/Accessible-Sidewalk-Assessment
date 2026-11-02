import torch.nn as nn
import torch.nn.functional as F


def masked_mean(loss, masks):
    return (loss * masks).sum() / masks.sum().clamp(min=1)


class MaskedBCELoss(nn.Module):
    """BCE over the tags valid for each image's category. Pass pos_weight for class-weighted BCE."""

    def __init__(self, pos_weight=None):
        super(MaskedBCELoss, self).__init__()
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, labels, masks):
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight, reduction='none')
        return masked_mean(loss, masks)


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(MaskedFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels, masks):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_correct = (-bce).exp()
        return masked_mean((1 - p_correct) ** self.gamma * bce, masks)
