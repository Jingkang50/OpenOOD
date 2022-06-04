from torch.utils.data import DataLoader

from openood.utils import Config

from .arpl_gan_trainer import ARPLGANTrainer
from .arpl_trainer import ARPLTrainer
from .base_trainer import BaseTrainer
from .conf_branch_trainer import ConfBranchTrainer
from .csi_trainer import CSITrainer
from .cutmix_trainer import CutMixTrainer
from .cutpaste_trainer import CutPasteTrainer
from .draem_trainer import DRAEMTrainer
from .dropout_trainer import DropoutTrainer
from .dsvdd_trainer import AETrainer, DSVDDTrainer
from .godin_trainer import GodinTrainer
from .kdad_trainer import KdadTrainer
from .mixup_trainer import MixupTrainer
from .mos_trainer import MOSTrainer
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
        'OpenMax': OpenMaxTrainer,
        'kdad': KdadTrainer,
        'conf_branch': ConfBranchTrainer,
        'dcae': AETrainer,
        'dsvdd': DSVDDTrainer,
        'openGan': OpenGanTrainer,
        'kdad': KdadTrainer,
        'godin': GodinTrainer,
        'arpl': ARPLTrainer,
        'arpl_gan': ARPLGANTrainer,
        'mos': MOSTrainer,
        'vos': VOSTrainer,
        'cutpaste': CutPasteTrainer,
        'cutmix': CutMixTrainer,
        'dropout': DropoutTrainer,
        'csi': CSITrainer,
    }
    return trainers[config.trainer.name](net, train_loader, config)
