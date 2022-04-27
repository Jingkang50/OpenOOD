from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .godin_postprocessor import GodinPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .patchcore_postprocessor import PatchcorePostprocessor
from .openmax_poseprocessor import OpenMax
from .react_postprocessor import ReactPostprocessor
from .vim_postprocessor import VIMPostprocessor
from openood.postprocessors.gradnorm_postprocessor import GradNormPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'ebo': EBOPostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'gmm': GMMPostprocessor,
        'Patchcore': PatchcorePostprocessor,
        'OpenMax': OpenMax,
        'react': ReactPostprocessor,
        'vim': VIMPostprocessor,
        'gradnorm': GradNormPostprocessor,
        'godin': GodinPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
