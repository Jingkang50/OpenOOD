from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .patch_preprocessor import PatchPreProcessor
from .pixmix_preprocessor import PixMixPreprocessor
from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(config: Config, split):
    train_preprocessors = {
        'base': BasePreprocessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'pixmix': PixMixPreprocessor,
    }
    test_preprocessors = {
        'base': TestStandardPreProcessor,
        'draem': DRAEMPreprocessor,
        'cutpaste': CutPastePreprocessor,
        'pixmix': TestStandardPreProcessor,
        'patch': PatchPreProcessor,
    }

    if split == 'train':
        return train_preprocessors[config.preprocessor.name](config, split)
    else:
        return test_preprocessors[config.preprocessor.name](config, split)
