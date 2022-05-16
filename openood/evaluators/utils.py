from openood.evaluators.openGan_evaluator import OpenGanEvaluator
from openood.utils import Config

from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .conf_esti_evaluator import Conf_Esti_Evaluator
from .draem_evaluator import DRAEMEvaluator
from .dsvdd_evaluator import DCAEEvaluator, DSVDDEvaluator
from .fsood_evaluator import FSOODEvaluator
from .kdad_evaluator import KdadDetectionEvaluator
from .ood_evaluator import OODEvaluator
from .vos_evaluator import VOSEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'fsood': FSOODEvaluator,
        'DRAEM': DRAEMEvaluator,
        'kdad': KdadDetectionEvaluator,
        'conf_esti': Conf_Esti_Evaluator,
        'openGan': OpenGanEvaluator,
        'kdad': KdadDetectionEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator,
        'arpl': ARPLEvaluator,
        'vos': VOSEvaluator
    }
    return evaluators[config.evaluator.name](config)
