from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .csi_trainer import CsiTrainer
from .draem_trainer import DRAEMTrainer
from .godin_trainer import GodinTrainer
from .kdad_trainer import KdadTrainer
from .mixup_trainer import MixupTrainer
from .opengan_trainer import OpenGanTrainer
from .sae_trainer import SAETrainer


def get_trainer(
    net,
    train_loader: DataLoader,
    config: Config,
):
    trainers = {
        'base': BaseTrainer,
        'mixup': MixupTrainer,
        'sae': SAETrainer,
        'draem': DRAEMTrainer,
        'opengan': OpenGanTrainer,
        'kdad': KdadTrainer,
        'godin': GodinTrainer,
        'csi': CsiTrainer,
    }
    return trainers[config.trainer.name](net, train_loader, config)
