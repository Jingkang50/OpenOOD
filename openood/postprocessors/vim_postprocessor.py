from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b = net.get_fc()
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net(data, return_feature=True)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                self.logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            
            self.feature_id_train = feature_id_train
            self.eigen_vectors = eigen_vectors
            self.eig_vals = eig_vals
            
            self._compute_NS_and_alpha()

            self.setup_flag = True
        else:
            pass
        
    def _compute_NS_and_alpha(self):
        """根据当前的dim计算NS和alpha"""
        self.NS = np.ascontiguousarray(
            (self.eigen_vectors.T[np.argsort(self.eig_vals * -1)[self.dim:]]).T)

        vlogit_id_train = norm(np.matmul(self.feature_id_train - self.u,
                                         self.NS),
                               axis=-1)
        self.alpha = self.logit_id_train.max(
            axis=-1).mean() / vlogit_id_train.mean()
        print(f'dim={self.dim}, {self.alpha=:.4f}')

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]
        if self.setup_flag:  
            self._compute_NS_and_alpha()

    def get_hyperparam(self):
        return self.dim