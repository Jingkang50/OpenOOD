from openood.utils import Config

from .base_recorder import BaseRecorder
from .conf_esti_recorder import Conf_Esti_Recorder
from .draem_recorder import DRAEMRecorder
from .kdad_recorder import KdadRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'DRAEM': DRAEMRecorder,
        'kdad': KdadRecorder,
        'conf_esti': Conf_Esti_Recorder
    }

    return recorders[config.recorder.name](config)
