import os.path as osp
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


class EnsemblePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(EnsemblePostprocessor, self).__init__(config)
        self.config = config
        self.postprocess_config = config.postprocessor
        self.postprocessor_args = self.postprocess_config.postprocessor_args
        assert self.postprocessor_args.network_name == \
            self.config.network.name,\
            'checkpoint network type and model type do not align!'
        # get ensemble args
        self.checkpoint_root = self.postprocessor_args.checkpoint_root

        # list of trained network checkpoints
        self.checkpoints = self.postprocessor_args.checkpoints
        # number of networks to esembel
        self.num_networks = self.postprocessor_args.num_networks
        # get networks
        self.checkpoint_dirs = [
            osp.join(self.checkpoint_root, path, 'best.ckpt')
            for path in self.checkpoints
        ]

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.networks = [deepcopy(net) for i in range(self.num_networks)]
        for i in range(self.num_networks):
            self.networks[i].load_state_dict(torch.load(
                self.checkpoint_dirs[i]),
                                             strict=False)
            self.networks[i].eval()

    def postprocess(self, net: nn.Module, data: Any):
        logits_list = [
            self.networks[i](data) for i in range(self.num_networks)
        ]
        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.num_networks):
            logits_mean += logits_list[i]
        logits_mean /= self.num_networks

        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
