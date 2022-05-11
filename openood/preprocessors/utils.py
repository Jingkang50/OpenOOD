from openood.preprocessors.pixmix_preprocessor import PixMixPreprocessor
from openood.utils import Config

from .base_preprocessor import BasePreprocessor
from .cutpaste_preprocessor import CutPastePreprocessor
from .draem_preprocessor import DRAEMPreprocessor
from .test_preprocessor import TestStandardPreProcessor
from .pixmix_preprocessor import PixMixPreprocessor
from .patch_preprocessor import PatchStandardPreProcessor
from .patch_gts_preprocessor import PatchGTStandardPreProcessor


def get_preprocessor(split, config: Config):
  train_preprocessors = {
    'base': BasePreprocessor,
    'DRAEM': DRAEMPreprocessor,
    'cutpaste': CutPastePreprocessor,
    'pixmix': PixMixPreprocessor, 
  }
  if split == "train":
    return train_preprocessors[config.preprocessor.name](config)    # need to pass in config.preprocessor
  elif split == 'patch' or split == 'patchTest' or split == 'patchTestGood':
    return PatchStandardPreProcessor(config)
  elif split == 'patchGT':
    return  PatchGTStandardPreProcessor(config)
  else:    # for test and val
    return TestStandardPreProcessor[split](split, config)



