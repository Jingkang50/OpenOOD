from openood.utils import Config

from .base_recorder import BaseRecorder
from .draem_recorder import DRAEMRecorder
from .dsvdd_recorder import DCAERecorder, DSVDDRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder
from .arpl_recorder import ARPLRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'DRAEM': DRAEMRecorder,
        'kdad': KdadRecorder,
        'dcae': DCAERecorder,
        'dsvdd': DSVDDRecorder,
        'openGan': OpenGanRecorder,
        'kdad': KdadRecorder,
        'arpl': ARPLRecorder,
    }

    return recorders[config.recorder.name](config)
