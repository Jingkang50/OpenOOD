from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .draem_evaluator import DRAEMEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator
from .openmax_evaluator import OpenMaxEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'DRAEM': DRAEMEvaluator,
        'openmax': OpenMaxEvaluator,
    }
    return evaluators[config.evaluator.name](config)
