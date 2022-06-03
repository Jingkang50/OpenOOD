from openood.utils import Config

from .ad_recorder import ADRecorder
from .arpl_recorder import ARPLRecorder
from .base_recorder import BaseRecorder
from .conf_esti_recorder import Conf_Esti_Recorder
from .cutpaste_recorder import CutpasteRecorder
from .draem_recorder import DRAEMRecorder
from .dsvdd_recorder import DCAERecorder, DSVDDRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'draem': DRAEMRecorder,
        'opengan': OpenGanRecorder,
        'conf_esti': Conf_Esti_Recorder,
        'dcae': DCAERecorder,
        'dsvdd': DSVDDRecorder,
        'kdad': KdadRecorder,
        'arpl': ARPLRecorder,
        'cutpaste': CutpasteRecorder,
        'ad': ADRecorder,
    }

    return recorders[config.recorder.name](config)
