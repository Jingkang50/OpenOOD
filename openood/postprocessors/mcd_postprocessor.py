from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class MCDPostprocessor(BasePostprocessor):
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits1, logits2 = net(data, return_double=True)
        score1 = torch.softmax(logits1, dim=1)
        score2 = torch.softmax(logits2, dim=1)
        conf = -torch.sum(torch.abs(score1 - score2), dim=1)
        _, pred = torch.max(score1, dim=1)
        return pred, conf
