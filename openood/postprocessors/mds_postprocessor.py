from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .base_postprocessor import BasePostprocessor
from .mds_tools import (alpha_selector, compute_noise_Mahalanobis_score,
                        get_Mahalanobis_score, sample_estimator)


class MDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.magnitude = self.postprocessor_args.noise
        self.feature_type_list = self.postprocessor_args.feature_type_list
        self.reduce_dim_list = self.postprocessor_args.reduce_dim_list

        self.num_classes = self.config.dataset.num_classes
        self.num_layer = len(self.feature_type_list)

        self.feature_mean, self.feature_prec = None, None
        self.alpha_list = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # step 1: estimate initial mean and variance from training set
        self.feature_mean, self.feature_prec = sample_estimator(
            net, id_loader_dict['train'], self.num_classes,
            self.feature_type_list, self.reduce_dim_list)

        # step 2: input process and hyperparam searching for alpha
        if self.postprocessor_args.alpha_list:
            print('Load predefined alpha list...')
            self.alpha_list = self.postprocessor_args.alpha_list
        else:
            print('Searching for optimal alpha list...')
            # get in-distribution scores
            for layer_index in range(self.num_layer):
                M_in = get_Mahalanobis_score(net, id_loader_dict['val'],
                                             self.num_classes,
                                             self.feature_mean,
                                             self.feature_prec, layer_index,
                                             self.magnitude)
                M_in = np.asarray(M_in, dtype=np.float32)
                if layer_index == 0:
                    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                else:
                    Mahalanobis_in = np.concatenate(
                        (Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))),
                        axis=1)
            # get out-of-distribution scores
            for layer_index in range(self.num_layer):
                M_out = get_Mahalanobis_score(net, ood_loader_dict['val'],
                                              self.num_classes,
                                              self.feature_mean,
                                              self.feature_prec, layer_index,
                                              self.magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if layer_index == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate(
                        (Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))),
                        axis=1)
            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)

            # logistic regression for optimal alpha
            self.alpha_list = alpha_selector(Mahalanobis_in, Mahalanobis_out)

    def postprocess(self, net: nn.Module, data: Any):
        for layer_index in range(self.num_layer):
            pred, score = compute_noise_Mahalanobis_score(
                net,
                Variable(data, requires_grad=True),
                self.num_classes,
                self.feature_mean,
                self.feature_prec,
                layer_index,
                self.magnitude,
                return_pred=True)
            if layer_index == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.cuda.FloatTensor(self.alpha_list)
        conf = torch.matmul(score_list, alpha)
        return pred, conf
