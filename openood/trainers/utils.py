from torch.utils.data import DataLoader

from openood.utils import Config

from .arpl_gan_trainer import ARPLGANTrainer
from .arpl_trainer import ARPLTrainer
from .augmix_trainer import AugMixTrainer
from .base_trainer import BaseTrainer
from .cider_trainer import CIDERTrainer
from .conf_branch_trainer import ConfBranchTrainer
from .csi_trainer import CSITrainer
from .cutmix_trainer import CutMixTrainer
from .cutpaste_trainer import CutPasteTrainer
from .draem_trainer import DRAEMTrainer
from .dropout_trainer import DropoutTrainer
from .dsvdd_trainer import AETrainer, DSVDDTrainer
from .godin_trainer import GodinTrainer
from .kdad_trainer import KdadTrainer
from .logitnorm_trainer import LogitNormTrainer
from .mcd_trainer import MCDTrainer
from .mixup_trainer import MixupTrainer
from .mos_trainer import MOSTrainer
from .npos_trainer import NPOSTrainer
from .oe_trainer import OETrainer
from .opengan_trainer import OpenGanTrainer
from .rd4ad_trainer import Rd4adTrainer
from .sae_trainer import SAETrainer
from .udg_trainer import UDGTrainer
from .vos_trainer import VOSTrainer
from .rts_trainer import RTSTrainer
from .rotpred_trainer import RotPredTrainer
from .regmixup_trainer import RegMixupTrainer
from .mixoe_trainer import MixOETrainer
from .ish_trainer import ISHTrainer
from .palm_trainer import PALMTrainer
from .t2fnorm_trainer import T2FNormTrainer
from .reweightood_trainer import ReweightOODTrainer
from .ascood_trainer import ASCOODTrainer


def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader,
                config: Config):
    if type(train_loader) is DataLoader:
        trainers = {
            'base': BaseTrainer,
            'augmix': AugMixTrainer,
            'mixup': MixupTrainer,
            'regmixup': RegMixupTrainer,
            'sae': SAETrainer,
            'draem': DRAEMTrainer,
            'kdad': KdadTrainer,
            'conf_branch': ConfBranchTrainer,
            'dcae': AETrainer,
            'dsvdd': DSVDDTrainer,
            'npos': NPOSTrainer,
            'opengan': OpenGanTrainer,
            'kdad': KdadTrainer,
            'godin': GodinTrainer,
            'arpl': ARPLTrainer,
            'arpl_gan': ARPLGANTrainer,
            'mos': MOSTrainer,
            'vos': VOSTrainer,
            'cider': CIDERTrainer,
            'cutpaste': CutPasteTrainer,
            'cutmix': CutMixTrainer,
            'dropout': DropoutTrainer,
            'csi': CSITrainer,
            'logitnorm': LogitNormTrainer,
            'rd4ad': Rd4adTrainer,
            'rts': RTSTrainer,
            'rotpred': RotPredTrainer,
            'ish': ISHTrainer,
            'palm': PALMTrainer,
            't2fnorm': T2FNormTrainer,
            'reweightood': ReweightOODTrainer,
            'ascood': ASCOODTrainer,
        }
        if config.trainer.name in ['cider', 'npos']:
            return trainers[config.trainer.name](net, train_loader, val_loader,
                                                 config)
        else:
            return trainers[config.trainer.name](net, train_loader, config)

    else:
        trainers = {
            'oe': OETrainer,
            'mcd': MCDTrainer,
            'udg': UDGTrainer,
            'mixoe': MixOETrainer
        }
        return trainers[config.trainer.name](net, train_loader[0],
                                             train_loader[1], config)
