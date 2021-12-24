from typing import Any, Dict

import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


def get_trainer(
    name: str,
    net: nn.Module,
    labeled_train_loader: DataLoader,
    optim_args: Dict[str, Any],
    trainer_args: Dict[str, Any],
):
    trainers = {
        'base': BaseTrainer,
    }

    return trainers[name](net, labeled_train_loader, **optim_args,
                          **trainer_args)
