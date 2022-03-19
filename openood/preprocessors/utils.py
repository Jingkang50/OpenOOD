from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutout_preprocessor import CutOutPreprocessor
from .scar_preprocessor import ScarPreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .cpscar_preprocessor import CPScarPreprocessor


def get_preprocessor(config: Config):
    preprocessors = {
        'base': BasePreprocessor,
        'cutpaste': CutPastePreprocessor,
        'cpscar': CPScarPreprocessor,
        'cutout': CutOutPreprocessor,
        'scar': ScarPreprocessor,
    }

    return preprocessors[config.preprocessor.name](config)
