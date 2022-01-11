from typing import Any

import torch
import torch.nn as nn


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def __call__(
        self,
        net: nn.Module,
        data: Any,
    ):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        return pred, conf
