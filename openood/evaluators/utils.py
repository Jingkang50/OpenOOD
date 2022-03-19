from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator
from .cutpaste_evaluator import CutPasteEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'cutpaste': CutPasteEvaluator,
    }
    return evaluators[config.evaluator.name](config)
