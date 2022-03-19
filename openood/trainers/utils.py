import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .cutpaste_trainer import CutPasteTrainer


def get_trainer(
    net: nn.Module,
    train_loader: DataLoader,
    config: Config,
):
    trainers = {
        'base': BaseTrainer,
        'cutpaste': CutPasteTrainer,
    }
    return trainers[config.trainer.name](net, train_loader, config)
