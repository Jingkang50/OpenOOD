from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from .base_postprocessor import BasePostprocessor

class VIMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.dim = self.args.dim

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        with torch.no_grad():
            w, b = net.get_fc() #TO_DO
            print("Extracting id training feature")
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()

                _, feature_id_train = net(data, return_feature=True)
                logit_id_train = feature_id_train @ w.T + b

            print("Extracting id testing feature")
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
            _, feature_id_val = net(data, return_feature=True)
            logit_id_val = feature_id_val @ w.T + b

        u = -np.matmul(pinv(w), b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        self.alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{self.alpha=:.4f}')

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * self.alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        self.score_id = -vlogit_id_val + energy_id_val

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net.forward_threshold(data, self.threshold)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
