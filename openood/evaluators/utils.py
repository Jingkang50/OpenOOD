from openood.evaluators.openGan_evaluator import OpenGanEvaluator
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .draem_evaluator import DRAEMEvaluator
from .dsvdd_evaluator import DCAEEvaluator, DSVDDEvaluator
from .fsood_evaluator import FSOODEvaluator
from .kdad_evaluator import KdadDetectionEvaluator
from .ood_evaluator import OODEvaluator
from .arpl_evaluator import ARPLEvaluator

def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'DRAEM': DRAEMEvaluator,
        'openGan': OpenGanEvaluator,
        'kdad': KdadDetectionEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator
        'arpl': ARPLEvaluator,
    }
    return evaluators[config.evaluator.name](config)
