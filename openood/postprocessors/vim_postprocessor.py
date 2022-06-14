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
        self.dim = self.args.dim

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        with torch.no_grad():
            self.w, self.b = net.get_fc()
            print('Extracting id training feature')
            feature_id_train = []
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                _, feature = net(data, return_feature=True)
                feature_id_train.append(feature.cpu().numpy())
            feature_id_train = np.concatenate(feature_id_train, axis=0)
            logit_id_train = feature_id_train @ self.w.T + self.b

            print('Extracting id testing feature')
            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                _, feature = net(data, return_feature=True)
                feature_id_val.append(feature.cpu().numpy())
            feature_id_val = np.concatenate(feature_id_val, axis=0)
            logit_id_val = feature_id_val @ self.w.T + self.b

        self.u = -np.matmul(pinv(self.w), self.b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

        # import pdb
        # pdb.set_trace()

        vlogit_id_train = norm(np.matmul(feature_id_train - self.u, self.NS),
                               axis=-1)
        self.alpha = logit_id_train.max(
            axis=-1).mean() / vlogit_id_train.mean()
        print(f'{self.alpha=:.4f}')

        vlogit_id_val = norm(np.matmul(feature_id_val - self.u, self.NS),
                             axis=-1) * self.alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        self.score_id = -vlogit_id_val + energy_id_val

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
