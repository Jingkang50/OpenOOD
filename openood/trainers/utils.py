from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .conf_esti_trainer import Conf_Esti_Trainer
from .draem_trainer import DRAEMTrainer
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
        'conf_esti': Conf_Esti_Trainer
    }
    return trainers[config.trainer.name](net, train_loader, config)
