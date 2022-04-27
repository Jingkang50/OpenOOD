from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .gram_postprocessor import GRAMPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .react_postprocessor import ReactPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'ebo': EBOPostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'gmm': GMMPostprocessor,
        'react': ReactPostprocessor,
        'gram': GRAMPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
