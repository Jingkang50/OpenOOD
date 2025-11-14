import os
import urllib.request

from openood.postprocessors import (
    AdaScalePostprocessor,
    ASHPostprocessor,
    BasePostprocessor,
    CIDERPostprocessor,
    ConfBranchPostprocessor,
    CutPastePostprocessor,
    DICEPostprocessor,
    DRAEMPostprocessor,
    DropoutPostProcessor,
    DSVDDPostprocessor,
    EBOPostprocessor,
    EnsemblePostprocessor,
    GENPostprocessor,
    GMMPostprocessor,
    GodinPostprocessor,
    GradNormPostprocessor,
    GRAMPostprocessor,
    GrOODPostprocessor,
    IODINPostprocessor,
    KLMatchingPostprocessor,
    KNNPostprocessor,
    MaxLogitPostprocessor,
    MCDPostprocessor,
    MDSEnsemblePostprocessor,
    MDSPostprocessor,
    MOSPostprocessor,
    NCIPostprocessor,
    NNGuidePostprocessor,
    NPOSPostprocessor,
    ODINPostprocessor,
    OpenGanPostprocessor,
    OpenMax,
    PatchcorePostprocessor,
    RankFeatPostprocessor,
    Rd4adPostprocessor,
    ReactPostprocessor,
    RelationPostprocessor,
    ResidualPostprocessor,
    ReweightOODPostprocessor,
    RMDSPostprocessor,
    RotPredPostprocessor,
    ScalePostprocessor,
    SHEPostprocessor,
    SSDPostprocessor,
    T2FNormPostprocessor,
    TemperatureScalingPostprocessor,
    VIMPostprocessor,
    fDBDPostprocessor,
)
from openood.utils.config import Config, merge_configs

postprocessors = {
    'nci': NCIPostprocessor,
    'fdbd': fDBDPostprocessor,
    'ash': ASHPostprocessor,
    'cider': CIDERPostprocessor,
    'conf_branch': ConfBranchPostprocessor,
    'msp': BasePostprocessor,
    'ebo': EBOPostprocessor,
    'odin': ODINPostprocessor,
    'iodin': IODINPostprocessor,
    'mds': MDSPostprocessor,
    'mds_ensemble': MDSEnsemblePostprocessor,
    'npos': NPOSPostprocessor,
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
    'scale': ScalePostprocessor,
    'ssd': SSDPostprocessor,
    'she': SHEPostprocessor,
    'rd4ad': Rd4adPostprocessor,
    'rotpred': RotPredPostprocessor,
    'rankfeat': RankFeatPostprocessor,
    'gen': GENPostprocessor,
    'nnguide': NNGuidePostprocessor,
    'relation': RelationPostprocessor,
    't2fnorm': T2FNormPostprocessor,
    'reweightood': ReweightOODPostprocessor,
    'adascale_a': AdaScalePostprocessor,
    'adascale_l': AdaScalePostprocessor,
    'grood': GrOODPostprocessor,
}

link_prefix = 'https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/'


def get_postprocessor(config_root: str, postprocessor_name: str,
                      id_data_name: str):
    postprocessor_config_path = os.path.join(config_root, 'postprocessors',
                                             f'{postprocessor_name}.yml')
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(
            link_prefix + f'{postprocessor_name}.yml',
            postprocessor_config_path,
        )

    config = Config(postprocessor_config_path)
    config = merge_configs(config,
                           Config(**{'dataset': {
                               'name': id_data_name
                           }}))
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor
