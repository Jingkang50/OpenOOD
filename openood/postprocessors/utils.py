from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .conf_branch_postprocessor import ConfBranchPostprocessor
from .cutpaste_postprocessor import CutPastePostprocessor
from .dice_postprocessor import DICEPostprocessor
from .draem_postprocessor import DRAEMPostprocessor
from .dropout_postprocessor import DropoutPostProcessor
from .dsvdd_postprocessor import DSVDDPostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .ensemble_postprocessor import EnsemblePostprocessor
from .gmm_postprocessor import GMMPostprocessor
from .godin_postprocessor import GodinPostprocessor
from .gradnorm_postprocessor import GradNormPostprocessor
from .gram_postprocessor import GRAMPostprocessor
from .kl_matching_postprocessor import KLMatchingPostprocessor
from .knn_postprocessor import KNNPostprocessor
from .maxlogit_postprocessor import MaxLogitPostprocessor
from .mcd_postprocessor import MCDPostprocessor
from .mds_postprocessor import MDSPostprocessor
from .mos_postprocessor import MOSPostprocessor
from .odin_postprocessor import ODINPostprocessor
from .opengan_postprocessor import OpenGanPostprocessor
from .openmax_postprocessor import OpenMax
from .patchcore_postprocessor import PatchcorePostprocessor
from .rd4ad_postprocessor import Rd4adPostprocessor
from .react_postprocessor import ReactPostprocessor
from .residual_postprocessor import ResidualPostprocessor
from .ssd_postprocessor import SSDPostprocessor
from .temp_scaling_postprocessor import TemperatureScalingPostprocessor
from .vim_postprocessor import VIMPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'conf_branch': ConfBranchPostprocessor,
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
        'mls': MaxLogitPostprocessor,
        'residual': ResidualPostprocessor,
        'klm': KLMatchingPostprocessor,
        'temperature_scaling': TemperatureScalingPostprocessor,
        'ensemble': EnsemblePostprocessor,
        'dropout': DropoutPostProcessor,
        'draem': DRAEMPostprocessor,
        'dsvdd': DSVDDPostprocessor,
        'mos': MOSPostprocessor,
        'mcd': MCDPostprocessor,
        'opengan': OpenGanPostprocessor,
        'knn': KNNPostprocessor,
        'dice': DICEPostprocessor,
        'ssd': SSDPostprocessor,
        'rd4ad': Rd4adPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
