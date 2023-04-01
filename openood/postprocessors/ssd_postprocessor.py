import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor
from .mds_ensemble_postprocessor import get_MDS_stat


class SSDPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args

        self.feature_type_list = self.postprocessor_args.feature_type_list
        self.reduce_dim_list = self.postprocessor_args.reduce_dim_list

        # self.num_classes = self.config.dataset.num_classes
        self.num_classes = 1
        self.num_layer = len(self.feature_type_list)

        self.feature_mean, self.feature_prec = None, None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.feature_mean, self.feature_prec, self.transform_matrix = \
            get_MDS_stat(net, id_loader_dict['train'], self.num_classes,
                         self.feature_type_list, self.reduce_dim_list)
