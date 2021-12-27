from typing import Any, Dict

from openood.utils import Config

from .base_recorder import BaseRecorder


def get_recorder(config: Config, ):
    recorders = {
        'base': BaseRecorder,
    }

    return recorders[config.recorder.name](config)
