from openood.utils import Config

from .base_recorder import BaseRecorder
from .draem_recorder import DRAEMRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'draem': DRAEMRecorder,
        'opengan': OpenGanRecorder,
        'kdad': KdadRecorder
    }

    return recorders[config.recorder.name](config)
