import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .draem_trainer import DRAEMTrainer
from .mixup_trainer import MixupTrainer
from .sae_trainer import SAETrainer
from .kdad_trainer import KdadTrainer

def get_trainer(
    net: nn.Module,
    train_loader: DataLoader,
    config: Config,
):
    trainers = {
        'base': BaseTrainer,
        'mixup': MixupTrainer,
        'sae': SAETrainer,
        'DRAEM': DRAEMTrainer,
        'kdad':KdadTrainer
    }
    return trainers[config.trainer.name](net, train_loader, config)
