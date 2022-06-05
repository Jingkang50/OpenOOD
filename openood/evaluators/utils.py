from openood.evaluators.mos_evaluator import MOSEvaluator
from openood.utils import Config

from .ad_evaluator import ADEvaluator
from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .fsood_evaluator import FSOODEvaluator
from .ood_evaluator import OODEvaluator
from .patchcore_evaluator import PatchCoreEvaluator
from .ece_evaluator import ECEEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'patch': PatchCoreEvaluator,
        'arpl': ARPLEvaluator,
        'ad': ADEvaluator,
        'mos': MOSEvaluator,
        'ece': ECEEvaluator,
    }
    return evaluators[config.evaluator.name](config)
