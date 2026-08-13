from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class PLFPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PLFPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        # 7 hyperparameters for PLF postprocessor
        self.y_start = self.args.y_start  # params 1
        self.y_end = self.args.y_end  # params 2
        self.q1 = self.args.q1
        self.x1 = None  # calculated in setup based on q1 - params 3
        self.y1 = None
        # calculated in setup based on y_end + delta_y - params 4
        self.m1 = self.args.m1  # params 5
        self.q2 = None  # calculated in setup based on q1 and u (delta)
        self.x2 = None  # calculated in setup based on q2 - params 6
        self.m2 = self.args.m2  # params 7
        self.delta_y = self.args.delta_y
        self.u = self.args.u

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

        self.y1 = self.y_end + self.delta_y
        delta_min = 0.10
        delta_max = 0.99 - self.q1
        delta = delta_min + self.u * (delta_max - delta_min)
        self.q2 = self.q1 + delta

        sorted_id_feats = np.sort(np.abs(self.activation_log), axis=0)
        n_id = sorted_id_feats.shape[0]

        i1 = int(self.q1 * (n_id - 1))
        i2 = int(self.q2 * (n_id - 1))
        if i2 <= i1:
            i2 = min(i1 + 1, n_id - 1)

        self.x1 = sorted_id_feats[i1]
        self.x2 = sorted_id_feats[i2]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.y_start, self.y_end, self.x1,
                                       self.y1, self.m1, self.x2, self.m2)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.y_start, self.y_end, self.delta_y, \
            self.q1, self.u, self.m1, self.m2 = hyperparam

        self.y1 = self.y_end + self.delta_y
        delta_min = 0.10
        delta_max = 0.99 - self.q1
        delta = delta_min + self.u * (delta_max - delta_min)
        self.q2 = self.q1 + delta

        sorted_id_feats = np.sort(np.abs(self.activation_log), axis=0)
        n_id = sorted_id_feats.shape[0]

        i1 = int(self.q1 * (n_id - 1))
        i2 = int(self.q2 * (n_id - 1))
        if i2 <= i1:
            i2 = min(i1 + 1, n_id - 1)

        self.x1 = sorted_id_feats[i1]
        self.x2 = sorted_id_feats[i2]

    def get_hyperparam(self):
        return self.y_start, self.y_end, self.delta_y, \
            self.q1, self.u, self.m1, self.m2
