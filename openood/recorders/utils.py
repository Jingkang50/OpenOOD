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
from .palm_recorder import PALMRecorder


def get_recorder(config: Config):
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
        'palm': PALMRecorder,
    }

    return recorders[config.recorder.name](config)
