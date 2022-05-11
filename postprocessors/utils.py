from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .cutpaste_postprocessor import CutPastePostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .godin_postprocessor import GodinPostprocessor
from .gradnorm_postprocessor import GradNormPostprocessor
from .gram_postprocessor import GRAMPostprocessor
from .kl_matching_postprocessor import KLMatchingPostprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .openmax_postprocessor import OpenMax
from .patchcore_postprocessor import PatchcorePostprocessor
from .react_postprocessor import ReactPostprocessor
from .residual_postprocessor import ResidualPostprocessor
from .vim_postprocessor import VIMPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'ebo': EBOPostprocessor,
        'odin': ODINPostprocessor,
        'mds': MDSPostprocessor,
        'gmm': GMMPostprocessor,
        'patchcore': PatchcorePostprocessor,
        'openmax': OpenMax,
        'react': ReactPostprocessor,
        'vim': VIMPostprocessor,
        'gradnorm': GradNormPostprocessor,
        'godin': GodinPostprocessor,
        'gram': GRAMPostprocessor,
        'cutpaste': CutPastePostprocessor,
        'maxlogit': MaxLogitPostprocessor,
        'residual': ResidualPostprocessor,
        'kl_matching': KLMatchingPostprocessor
    }

    return postprocessors[config.postprocessor.name](config)
