from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .draem_preprocessor import DRAEMPreprocessor


def get_preprocessor(config: Config):
    preprocessors = {
        'base': BasePreprocessor,
        'DRAEM': DRAEMPreprocessor
    }

    return preprocessors[config.preprocessor.name](config)
