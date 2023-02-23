from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class RTSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RTSPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.ood_score = self.args.ood_score

    def postprocess(self, net: nn.Module, data: Any):
        output, variance = net(data, return_var=True)
        if self.ood_score == 'var':
            _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
            conf = torch.mean(variance, dim=1)
        elif self.ood_score == 'msp':
            score = torch.softmax(output, dim=1)
            conf, pred = torch.max(score, dim=1)
        else:
            print('Invalid ood score type, using var instead')
            _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
            conf = torch.mean(variance, dim=1)
        return pred, conf
