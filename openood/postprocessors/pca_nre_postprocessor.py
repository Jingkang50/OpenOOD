from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class PCANREPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.k = self.args.k
        self.percentile = self.args.percentile
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                self.w, self.b = net.get_fc()
                for batch in tqdm(id_loader_dict['train'],
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

        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

        self.activation_log_mean = np.mean(self.activation_log, axis=0)

        cov = np.cov(self.activation_log.T)
        u, s, v = np.linalg.svd(cov)

        self.M = u[:, :self.k] @ u[:, :self.k].T
        self.dim = self.M.shape[0]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        feature_ood = feature_ood.clip(max=self.threshold)
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        rec_ood = np.linalg.norm((feature_ood - self.activation_log_mean)
                                 @ (np.identity(self.dim) - self.M),
                                 axis=-1)
        r_ood = rec_ood / np.linalg.norm(feature_ood, axis=-1)
        score_ood = logsumexp(logit_ood, axis=-1) * (1.0 - r_ood)
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.k = hyperparam[0]
        self.percentile = hyperparam[1]

    def get_hyperparam(self):
        return [self.k, self.percentile]
