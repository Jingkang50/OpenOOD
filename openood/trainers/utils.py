from torch.utils.data import DataLoader

from openood.utils import Config

from .arpl_gan_trainer import ARPLGANTrainer
from .arpl_trainer import ARPLTrainer
from .base_trainer import BaseTrainer
from .conf_esti_trainer import Conf_Esti_Trainer
from .csi_trainer import CsiTrainer
from .cutmix_trainer import CutMixTrainer
from .cutpaste_trainer import CutPasteTrainer
from .draem_trainer import DRAEMTrainer
from .dsvdd_trainer import AETrainer, DSVDDTrainer
from .godin_trainer import GodinTrainer
from .kdad_trainer import KdadTrainer
from .mixup_trainer import MixupTrainer
from .opengan_trainer import OpenGanTrainer
from .openmax_trainer import OpenMaxTrainer
from .sae_trainer import SAETrainer
from .vos_trainer import VOSTrainer



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
        'godin': GodinTrainer,
        'csi': CsiTrainer,
        'OpenMax': OpenMaxTrainer,
        'conf_esti': Conf_Esti_Trainer,
        'dcae': AETrainer,
        'dsvdd': DSVDDTrainer,
        'kdad': KdadTrainer,
        'arpl': ARPLTrainer,
        'arpl_gan': ARPLGANTrainer,
        'vos': VOSTrainer,
        'cutpaste': CutPasteTrainer,
        'cutmix': CutMixTrainer,
    }
    return trainers[config.trainer.name](net, train_loader, config)
