from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class ScalePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ScalePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.percentile)
        _, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile
