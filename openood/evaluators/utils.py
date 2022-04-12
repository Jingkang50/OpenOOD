from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .draem_evaluator import DRAEMEvaluator
from .dsvdd_evaluator import DCAEEvaluator, DSVDDEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'DRAEM': DRAEMEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator
    }
    return evaluators[config.evaluator.name](config)
