import torch

from .focal import FocalLoss
from .ssim import SSIM


def get_draem_losses():
    losses = {
        'l2': torch.nn.modules.loss.MSELoss(),
        'ssim': SSIM(),
        'focal': FocalLoss()
    }
    return losses
