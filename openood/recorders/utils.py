
from openood.recorders.recorder import RecorderProtocol
from openood.utils import Config

from .ad_recorder import ADRecorder
from .arpl_recorder import ARPLRecorder
from .base_recorder import BaseRecorder
from .cider_recorder import CiderRecorder
from .cutpaste_recorder import CutpasteRecorder
from .draem_recorder import DRAEMRecorder
from .dsvdd_recorder import DCAERecorder, DSVDDRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder
from .rd4ad_recorder import Rd4adRecorder
from .wandb_wrapper import WandbWrapper


def get_recorder(config: Config) -> RecorderProtocol:
    recorders = {
        'base': BaseRecorder,
        'cider': CiderRecorder,
        'draem': DRAEMRecorder,
        'opengan': OpenGanRecorder,
        'dcae': DCAERecorder,
        'dsvdd': DSVDDRecorder,
        'kdad': KdadRecorder,
        'arpl': ARPLRecorder,
        'cutpaste': CutpasteRecorder,
        'ad': ADRecorder,
        'rd4ad': Rd4adRecorder,
    }


    base_recorder = recorders[config.recorder.name.removeprefix('wandb_')](config)

    if config.recorder.name.startswith('wandb_'):
        return WandbWrapper(recorder=base_recorder)
    else:     
        return base_recorder
