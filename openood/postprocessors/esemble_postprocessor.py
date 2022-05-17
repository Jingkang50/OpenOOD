from .base_postprocessor import BasePostprocessor
from torch import nn, optim
import torch
from typing import Any
import os.path as osp
from copy import deepcopy

class EsemblePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(EsemblePostprocessor, self).__init__(config)
        self.config = config
        self.postprocess_config = config.postprocessor
        self.postprocessor_args = self.postprocess_config.postprocessor_args
        assert self.postprocessor_args.network_name == self.config.network.name, \
            "checkpoint network type and model type do not align!"  
        # get esemble args
        self.checkpoint_root = self.postprocessor_args.checkpoint_root
        self.checkpoints = self.postprocessor_args.checkpoints   # list of trained network checkpoints
        self.num_networks = self.postprocessor_args.num_networks  # number of networks to esembel
        # get networks
        self.checkpoint_dirs = [osp.join(self.checkpoint_root, path, 'best.ckpt') for path in self.checkpoints]


    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.networks = [deepcopy(net) for i in range(self.num_networks)]
        for i in range(self.num_networks):
            self.networks[i].load_state_dict(torch.load(self.checkpoint_dirs[i]), strict = False)
    
    def postprocess(self, net: nn.Module, data: Any):
        logits_list = [self.networks[i](data) for i in range(self.num_networks)]
        logits_mean = torch.zeros_like(logits_list[0], dtype = torch.float32)
        for i in range(self.num_networks):
            logits_mean += logits_list[i]
        logits_mean /= self.num_networks
        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
