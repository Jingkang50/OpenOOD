import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer


class T2FNormTrainer(BaseTrainer):
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        super(T2FNormTrainer, self).__init__(net, train_loader, config)
        self.net.set_tau(config.trainer.trainer_args.tau)
