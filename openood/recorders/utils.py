from openood.utils import Config

from .base_recorder import BaseRecorder
from .cutpaste_recorder import CutpasteRecorder
from .draem_recorder import DRAEMRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'DRAEM': DRAEMRecorder,
        'openGan': OpenGanRecorder,
        'kdad': KdadRecorder,
        'cutpaste': CutpasteRecorder,
    }

    return recorders[config.recorder.name](config)
