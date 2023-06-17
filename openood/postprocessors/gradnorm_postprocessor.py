from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class GradNormPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]

    def gradnorm(self, x, w, b):
        fc = torch.nn.Linear(*w.shape[::-1])
        fc.weight.data[...] = torch.from_numpy(w)
        fc.bias.data[...] = torch.from_numpy(b)
        fc.cuda()

        targets = torch.ones((1, self.num_classes)).cuda()

        confs = []
        for i in x:
            fc.zero_grad()
            loss = torch.mean(
                torch.sum(-targets * F.log_softmax(fc(i[None]), dim=-1),
                          dim=-1))
            loss.backward()
            layer_grad_norm = torch.sum(torch.abs(
                fc.weight.grad.data)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        w, b = net.get_fc()
        logits, features = net.forward(data, return_feature=True)
        with torch.enable_grad():
            scores = self.gradnorm(features, w, b)
        _, preds = torch.max(logits, dim=1)
        return preds, torch.from_numpy(scores)
