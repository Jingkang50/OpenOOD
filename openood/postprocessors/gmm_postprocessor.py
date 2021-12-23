from typing import Any

import torch
import torch.nn as nn
from traitlets.traitlets import Bool

from .base_postprocessor import BasePostprocessor
from .gmm_tools import calculate_prob
from .mds_tools import process_feature_type


class GMMPostprocessor(BasePostprocessor):
    def __init__(self, feature_type_list, feature_mean_list, feature_prec_list,
                 component_weight_list, transform_matrix_list, alpha_list):
        super().__init__()
        self.feature_type_list = feature_type_list
        self.feature_mean_list = feature_mean_list
        self.feature_prec_list = feature_prec_list
        self.component_weight_list = component_weight_list
        self.transform_matrix_list = transform_matrix_list
        self.alpha_list = alpha_list

    @torch.no_grad()
    def __call__(self, net: nn.Module, data: Any, return_scores: Bool = False):
        output, feature_list = net(data, return_feature_list=True)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)

        for layer_idx in range(len(feature_list)):
            feature = process_feature_type(feature_list[layer_idx],
                                           self.feature_type_list[layer_idx])
            feature = torch.mm(feature, self.transform_matrix_list[layer_idx])
            score = calculate_prob(feature, self.feature_mean_list[layer_idx],
                                   self.feature_prec_list[layer_idx],
                                   self.component_weight_list[layer_idx])
            if layer_idx == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)

        if return_scores:
            return score_list

        else:
            alpha_list = torch.cuda.FloatTensor(self.alpha_list)
            conf = torch.matmul(score_list, alpha_list)
            return pred, conf
