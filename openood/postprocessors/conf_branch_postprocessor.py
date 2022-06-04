from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class ConfBranchPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ConfBranchPostprocessor, self).__init__(config)
        self.config = config

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, conf = net(data, return_confidence=True)
        conf = torch.sigmoid(conf)
        _, pred = torch.max(output, dim=1)
        return pred, conf
