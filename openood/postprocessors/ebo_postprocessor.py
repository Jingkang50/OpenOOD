from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class EBOPostprocessor(BasePostprocessor):
    @torch.no_grad()
    def __call__(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)

        return pred, conf
