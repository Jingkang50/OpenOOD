from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class RASPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RASPostprocessor, self).__init__(config)
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                loader = id_loader_dict['train']
                for batch in tqdm(loader,
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu())

            activation_log = torch.cat(activation_log, dim=0)
            sorted_vals, _ = torch.sort(activation_log, dim=1)
            mean_curve = sorted_vals.mean(dim=0).cuda()

            net.set_mean_curve(mean_curve)
            self.mean_curve = mean_curve
            self.setup_flag = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_shift(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, conf
