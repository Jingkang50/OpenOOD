from typing import Any

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from .base_postprocessor import BasePostprocessor


class NCIPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NCIPostprocessor, self).__init__(config)
        self.APS_mode = True
        self.setup_flag = False
        self.train_mean = None
        self.w = None
        self.activation_log = None

        self.args = self.config.postprocessor.postprocessor_args
        self.alpha = self.args.alpha
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # collect training mean
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)

                    activation_log.append(feature.data.cpu().numpy())

            activation_log_concat = np.concatenate(activation_log, axis=0)
            self.activation_log = activation_log_concat
            self.train_mean = torch.from_numpy(
                np.mean(activation_log_concat, axis=0)).cuda()

            # compute denominator matrix
            for i, param in enumerate(net.fc.parameters()):
                if i == 0:
                    self.w = param.data

            self.setup_flag = True

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        values, nn_idx = output.max(1)
        score = torch.sum(self.w[nn_idx] * (feature - self.train_mean), axis=1) /torch.norm(feature - self.train_mean, dim = 1) + self.alpha * torch.norm(feature, p = 1, dim = 1)
        return nn_idx, score

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]

    def get_hyperparam(self):
        return self.alpha
