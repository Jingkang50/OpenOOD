from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .kdad_trainer import KdadTrainer


def get_trainer(
    net,
    train_loader: DataLoader,
    config: Config,
):
    trainers = {'base': BaseTrainer, 'kdad': KdadTrainer}
    return trainers[config.trainer.name](net, train_loader, config)
