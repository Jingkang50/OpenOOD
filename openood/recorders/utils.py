from openood.utils import Config

from .base_recorder import BaseRecorder
from .draem_recorder import DRAEMRecorder
from .dsvdd_recorder import DCAERecorder, DSVDDRecorder
from .kdad_recorder import KdadRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'DRAEM': DRAEMRecorder,
        'kdad': KdadRecorder,
        'dcae': DCAERecorder,
        'dsvdd': DSVDDRecorder
    }

    return recorders[config.recorder.name](config)
