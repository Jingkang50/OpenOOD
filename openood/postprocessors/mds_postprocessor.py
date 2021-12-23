from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_postprocessor import BasePostprocessor
from .mds_tools import compute_noise_Mahalanobis_score


class MDSPostprocessor(BasePostprocessor):
    def __init__(self, feature_mean, feature_prec, magnitude, alpha_optimal):
        super().__init__()
        self.num_layer = len(feature_mean)
        self.num_classes = len(feature_mean[0])
        self.feature_mean = feature_mean
        self.feature_prec = feature_prec
        self.magnitude = magnitude
        self.alpha_optimal = alpha_optimal

    # @torch.no_grad()
    def __call__(
        self,
        net: nn.Module,
        data: Any,
    ):
        for layer_index in range(1, self.num_layer):
            pred, score = compute_noise_Mahalanobis_score(
                net,
                Variable(data, requires_grad=True),
                self.num_classes,
                self.feature_mean,
                self.feature_prec,
                layer_index,
                self.magnitude,
                return_pred=True)
            if layer_index == 1:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.cuda.FloatTensor(self.alpha_optimal)
        conf = torch.matmul(score_list, alpha)
        return pred, conf
