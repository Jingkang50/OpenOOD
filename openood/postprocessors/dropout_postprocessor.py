from .base_postprocessor import BasePostprocessor
from torch import nn
from typing import Any
import torch


class DropoutPostProcessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.args = config.postprocessor.postprocessor_args
        self.p = self.args.dropout_p
        self.dropout_times = self.args.dropout_times

    def postprocess(self, net: nn.Module, data: Any):
        with torch.no_grad():
            logits_list = [net.forward_with_dropout(data, self.p) for i in range(self.dropout_times)]
            logits_mean = torch.zeros_like(logits_list[0], dtype = torch.float32)
            for i in range(self.dropout_times):
                logits_mean += logits_list[i]
            logits_mean /= self.dropout_times
            score = torch.softmax(logits_mean, dim=1)
            conf, pred = torch.max(score, dim=1)
            return pred, conf

