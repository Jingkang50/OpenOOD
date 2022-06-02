from openood.utils import Config

from .ad_evaluator import ADEvaluator
from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .conf_esti_evaluator import Conf_Esti_Evaluator
from .cutpaste_evaluator import CutPasteEvaluator
from .draem_evaluator import DRAEMEvaluator
from .dsvdd_evaluator import DCAEEvaluator, DSVDDEvaluator
from .fsood_evaluator import FSOODEvaluator
from .kdad_evaluator import KdadDetectionEvaluator
from .ood_evaluator import OODEvaluator
from .openmax_evaluator import OpenMaxEvaluator
from .patchcore_evaluator import PatchCoreEvaluator
from .vos_evaluator import VOSEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'draem': DRAEMEvaluator,
        'openmax': OpenMaxEvaluator,
        'kdad': KdadDetectionEvaluator,
        'conf_esti': Conf_Esti_Evaluator,
        'patch': PatchCoreEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator,
        'arpl': ARPLEvaluator,
        'vos': VOSEvaluator,
        'cutpaste': CutPasteEvaluator,
        'ad': ADEvaluator,
    }
    return evaluators[config.evaluator.name](config)
