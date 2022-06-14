from typing import Any

import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


class DropoutPostProcessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args
        self.dropout_times = self.args.dropout_times

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits_list = [net.forward(data) for i in range(self.dropout_times)]
        logits_mean = torch.zeros_like(logits_list[0], dtype=torch.float32)
        for i in range(self.dropout_times):
            logits_mean += logits_list[i]
        logits_mean /= self.dropout_times
        score = torch.softmax(logits_mean, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
