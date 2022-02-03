from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from .gmm_tools import compute_GMM_score, get_GMM_stat


class GMMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.feature_type_list = self.postprocessor_args.feature_type_list
        self.reduce_dim_list = self.postprocessor_args.reduce_dim_list
        self.num_clusters_list = self.postprocessor_args.num_clusters_list
        self.alpha_list = self.postprocessor_args.alpha_list

        self.num_layer = len(self.feature_type_list)
        self.feature_mean, self.feature_prec = None, None
        self.component_weight_list, self.transform_matrix_list = None, None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.feature_mean, self.feature_prec, self.component_weight_list, \
            self.transform_matrix_list = get_GMM_stat(net,
                                                      id_loader_dict['train'],
                                                      self.num_clusters_list,
                                                      self.feature_type_list,
                                                      self.reduce_dim_list)

    def postprocess(self, net: nn.Module, data: Any):
        for layer_index in range(self.num_layer):
            pred, score = compute_GMM_score(net,
                                            data,
                                            self.feature_mean,
                                            self.feature_prec,
                                            self.component_weight_list,
                                            self.transform_matrix_list,
                                            layer_index,
                                            self.feature_type_list,
                                            return_pred=True)
            if layer_index == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.cuda.FloatTensor(self.alpha_list)
        conf = torch.matmul(score_list, alpha)
        return pred, conf
