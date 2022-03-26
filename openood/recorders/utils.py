from typing import Any, Dict

from openood.utils import Config

from .base_recorder import BaseRecorder
from .cutpaste_recorder import CutpasteRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
        'cutpaste': CutpasteRecorder,
    }

    return recorders[config.recorder.name](config)
