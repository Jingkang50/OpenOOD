import os

from openood.postprocessors import (
    ASHPostprocessor, BasePostprocessor, ConfBranchPostprocessor,
    CutPastePostprocessor, DICEPostprocessor, DRAEMPostprocessor,
    DropoutPostProcessor, DSVDDPostprocessor, EBOPostprocessor,
    EnsemblePostprocessor, GMMPostprocessor, GodinPostprocessor,
    GradNormPostprocessor, GRAMPostprocessor, KLMatchingPostprocessor,
    KNNPostprocessor, MaxLogitPostprocessor, MCDPostprocessor,
    MDSPostprocessor, MDSEnsemblePostprocessor, MOSPostprocessor,
    ODINPostprocessor, OpenGanPostprocessor, OpenMax, PatchcorePostprocessor,
    Rd4adPostprocessor, ReactPostprocessor, ResidualPostprocessor,
    SSDPostprocessor, TemperatureScalingPostprocessor, VIMPostprocessor,
    RotPredPostprocessor, RankFeatPostprocessor, RMDSPostprocessor,
    SHEPostprocessor)
from openood.utils.config import Config, merge_configs

postprocessors = {
    'ash': ASHPostprocessor,
    'conf_branch': ConfBranchPostprocessor,
    'msp': BasePostprocessor,
    'ebo': EBOPostprocessor,
    'odin': ODINPostprocessor,
    'mds': MDSPostprocessor,
    'mds_ensemble': MDSEnsemblePostprocessor,
    'rmds': RMDSPostprocessor,
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
    'she': SHEPostprocessor,
    'rd4ad': Rd4adPostprocessor,
    'rotpred': RotPredPostprocessor,
    'rankfeat': RankFeatPostprocessor
}


def get_postprocessor(config_root: str, postprocessor_name: str,
                      num_classes: int, std: list):
    config = Config(
        os.path.join(config_root, 'postprocessors',
                     f'{postprocessor_name}.yml'))
    config = merge_configs(
        config, Config(**{'dataset': {
            'num_classes': num_classes,
            'std': std
        }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
