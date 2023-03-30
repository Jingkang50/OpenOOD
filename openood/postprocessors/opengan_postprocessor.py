from typing import Any

import torch

from .base_postprocessor import BasePostprocessor


class OpenGanPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(OpenGanPostprocessor, self).__init__(config)

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        # images input
        if data.shape[-1] > 1 and data.shape[1] == 3:
            output = net['backbone'](data)
            score = torch.softmax(output, dim=1)
            _, pred = torch.max(score, dim=1)

            _, feats = net['backbone'](data, return_feature=True)
            feats = feats.unsqueeze_(-1).unsqueeze_(-1)
            predConf = net['netD'](feats)
            predConf = predConf.view(-1, 1)
            conf = predConf.reshape(-1).detach().cpu()
        # feature input
        elif data.shape[-1] == 1 and data.shape[-1] == 1:
            predConf = net['netD'](data)
            predConf = predConf.view(-1, 1)
            conf = predConf.reshape(-1).detach().cpu()
            pred = torch.ones_like(conf)  # dummy predictions

        return pred, conf
