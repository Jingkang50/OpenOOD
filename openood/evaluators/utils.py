from openood.utils import Config

from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .conf_esti import Conf_Esti_Evaluator
from .cutpaste_evaluator import CutPasteEvaluator
from .draem_evaluator import DRAEMEvaluator
from .dsvdd_evaluator import DCAEEvaluator, DSVDDEvaluator
from .fsood_evaluator import FSOODEvaluator
from .kdad_evaluator import KdadDetectionEvaluator
from .ood_evaluator import OODEvaluator
from .opengan_evaluator import OpenGanEvaluator
from .openmax_evaluator import OpenMaxEvaluator
from .patchcore_evaluator import PatchCoreEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'draem': DRAEMEvaluator,
        'opengan': OpenGanEvaluator,
        'openmax': OpenMaxEvaluator,
        'kdad': KdadDetectionEvaluator,
        'conf_esti': Conf_Esti_Evaluator,
        'patch': PatchCoreEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator,
        'arpl': ARPLEvaluator,
        'cutpaste': CutPasteEvaluator,
    }
    return evaluators[config.evaluator.name](config)
