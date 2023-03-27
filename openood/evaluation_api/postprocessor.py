import os

from openood.postprocessors import (
    BasePostprocessor, ConfBranchPostprocessor, CutPastePostprocessor,
    DICEPostprocessor, DRAEMPostprocessor, DropoutPostProcessor,
    DSVDDPostprocessor, EBOPostprocessor, EnsemblePostprocessor,
    GMMPostprocessor, GodinPostprocessor, GradNormPostprocessor,
    GRAMPostprocessor, KLMatchingPostprocessor, KNNPostprocessor,
    MaxLogitPostprocessor, MCDPostprocessor, MDSPostprocessor,
    MOSPostprocessor, ODINPostprocessor, OpenGanPostprocessor, OpenMax,
    PatchcorePostprocessor, Rd4adPostprocessor, ReactPostprocessor,
    ResidualPostprocessor, SSDPostprocessor, TemperatureScalingPostprocessor,
    VIMPostprocessor)
from openood.utils.config import Config, merge_configs

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
    'mds': MDSPostprocessor,
    'gram': GRAMPostprocessor,
    'cutpaste': CutPastePostprocessor,
    'mls': MaxLogitPostprocessor,
    'residual': ResidualPostprocessor,
    'klm': KLMatchingPostprocessor,
    'temp_scaling': TemperatureScalingPostprocessor,
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


def get_postprocessor(config_root: str, postprocessor_name: str,
                      num_classes: int):
    config = Config(
        os.path.join(config_root, 'postprocessors',
                     f'{postprocessor_name}.yml'))
    config = merge_configs(config,
                           Config(**{'dataset': {
                               'num_classes': num_classes
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    return postprocessor
