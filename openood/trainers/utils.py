from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .cutpaste_trainer import CutPasteTrainer
from .draem_trainer import DRAEMTrainer
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
        'DRAEM': DRAEMTrainer,
        'openGan': OpenGanTrainer,
        'kdad': KdadTrainer,
        'cutpaste': CutPasteTrainer
    }
    return trainers[config.trainer.name](net, train_loader, config)
