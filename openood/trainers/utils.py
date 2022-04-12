from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .draem_trainer import DRAEMTrainer
from .dsvdd_trainer import AETrainer, DSVDDTrainer
from .kdad_trainer import KdadTrainer
from .mixup_trainer import MixupTrainer
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
        'kdad': KdadTrainer,
        'dcae': AETrainer,
        'dsvdd': DSVDDTrainer
    }
    return trainers[config.trainer.name](net, train_loader, config)
