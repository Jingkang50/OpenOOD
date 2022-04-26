from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .conf_esti import Conf_Esti_Evaluator
from .draem_evaluator import DRAEMEvaluator
from .fsood_evaluator import FSOODEvaluator
from .kdad_evaluator import KdadDetectionEvaluator
from .ood_evaluator import OODEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'DRAEM': DRAEMEvaluator,
        'kdad': KdadDetectionEvaluator,
        'conf_esti': Conf_Esti_Evaluator
    }
    return evaluators[config.evaluator.name](config)
