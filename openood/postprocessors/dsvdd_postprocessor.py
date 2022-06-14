from typing import Any

import torch
import torch.nn as nn

from openood.trainers.dsvdd_trainer import init_center_c

from .base_postprocessor import BasePostprocessor


class DSVDDPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DSVDDPostprocessor, self).__init__(config)
        self.hyperpara = {}

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.config.c == 'None' and self.config.network.name != 'dcae':
            self.c = init_center_c(id_loader_dict['train'], net)
        else:
            self.c = self.config.c

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        outputs = net(data)
        if self.config.network.name != 'dcae':
            conf = torch.sum((outputs - self.c)**2,
                             dim=tuple(range(1, outputs.dim())))

        # this is for pre-training the dcae network from the original paper
        elif self.config.network.name == 'dcae':
            conf = torch.sum((outputs - data)**2,
                             dim=tuple(range(1, outputs.dim())))
        else:
            raise NotImplementedError

        return -1 * torch.ones(data.shape[0]), conf
