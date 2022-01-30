import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
<<<<<<< HEAD
from .mixup_trainer import MixupTrainer
from .sae_trainer import SAETrainer
=======
from .draem_trainer import DRAEMTrainer
>>>>>>> dc67651 (add DRAEM ver_0)


def get_trainer(
    net: nn.Module,
    train_loader: DataLoader,
    config: Config,
):
    trainers = {
        'base': BaseTrainer,
<<<<<<< HEAD
        'mixup': MixupTrainer,
        'sae': SAETrainer,
=======
        'DRAEM': DRAEMTrainer,
>>>>>>> dc67651 (add DRAEM ver_0)
    }
    return trainers[config.trainer.name](net, train_loader, config)
