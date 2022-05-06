from openood.postprocessors import residual_postprocessor
from openood.postprocessors.kl_matching_postprocessor import KLMatchingPostprocessor
from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .cutpaste_postprocessor import CutPastePostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .godin_postprocessor import GodinPostprocessor
from .gradnorm_postprocessor import GradNormPostprocessor
from .gram_postprocessor import GRAMPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .openmax_postprocessor import OpenMax
from .patchcore_postprocessor import PatchcorePostprocessor
from .react_postprocessor import ReactPostprocessor
from .vim_postprocessor import VIMPostprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .residual_postprocessor import ResidualPostprocessor
from .kl_matching_postprocessor import KLMatchingPostprocessor

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
