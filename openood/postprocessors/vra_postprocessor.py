from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class VRAPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super(VRAPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile_high = self.args.percentile_high
        self.percentile_low = self.args.percentile_low
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass

        self.threshold_high = np.percentile(self.activation_log.flatten(),
                                            self.percentile_high)
        self.threshold_low = np.percentile(self.activation_log.flatten(),
                                           self.percentile_low)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.clip(min=self.threshold_low,
                                       max=self.threshold_high)
        feature_ood = feature_ood.view(feature_ood.size(0), -1)
        logit_ood = net.fc(feature_ood)
        score = torch.softmax(logit_ood, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(logit_ood.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.percentile_high = hyperparam[0]
        self.percentile_low = hyperparam[1]
        self.threshold_high = np.percentile(self.activation_log.flatten(),
                                            self.percentile_high)
        self.threshold_low = np.percentile(self.activation_log.flatten(),
                                           self.percentile_low)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile_high, self.threshold_high))
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile_low, self.threshold_low))

    def get_hyperparam(self):
        return [self.percentile_high, self.percentile_low]
