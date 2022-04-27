from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .transform import TestStandard, TrainStandard


def get_preprocessor(config: Config):
    preprocessors = {
        'base': BasePreprocessor,
        'DRAEM': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
    }

    return preprocessors[config.preprocessor.name](config)
