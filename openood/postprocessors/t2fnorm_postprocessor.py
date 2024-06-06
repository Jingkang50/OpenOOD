from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class T2FNormPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_ood_inference(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
