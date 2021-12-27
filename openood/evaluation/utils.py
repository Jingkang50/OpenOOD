import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_evaluator import BaseEvaluator


def get_evaluator(config: Config, ):
    evaluators = {
        'base': BaseEvaluator,
    }
    return evaluators[config.evaluator.name](config)
