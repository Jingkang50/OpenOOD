from typing import Any, Dict

from openood.utils import Config

from .base_recorder import BaseRecorder


def get_recorder(
    name: str,
    config: Config,
):
    recorders = {
        'base': BaseRecorder,
    }

    return recorders[name](config)
