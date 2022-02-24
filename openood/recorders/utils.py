from openood.utils import Config

from .base_recorder import BaseRecorder
from .draem_recorder import DRAEMRecorder
from .kdad_recorder import KdadRecorder

def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'DRAEM': DRAEMRecorder,
        'kdad': KdadRecorder
    }

    return recorders[config.recorder.name](config)
