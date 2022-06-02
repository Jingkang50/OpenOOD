from openood.evaluators.mos_evaluator import MOSEvaluator
from openood.utils import Config

from .arpl_evaluator import ARPLEvaluator
from .base_evaluator import BaseEvaluator
from .conf_branch_evaluator import ConfBranchEvaluator
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
        'confbranch': ConfBranchEvaluator,
        'patch': PatchCoreEvaluator,
        'dcae': DCAEEvaluator,
        'dsvdd': DSVDDEvaluator,
        'arpl': ARPLEvaluator,
        'mos': MOSEvaluator,
        'cutpaste': CutPasteEvaluator,
        'vos': VOSEvaluator,
    }
    return evaluators[config.evaluator.name](config)
