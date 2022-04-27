from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from .base_postprocessor import BasePostprocessor

class GradNormPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    def gradnorm(x, w, b):
        fc = torch.nn.Linear(*w.shape[::-1])
        fc.weight.data[...] = torch.from_numpy(w)
        fc.bias.data[...] = torch.from_numpy(b)
        fc.cuda()

        x = torch.from_numpy(x).float().cuda()
        logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

        confs = []

        for i in tqdm(x):
            targets = torch.ones((1, 1000)).cuda()
            fc.zero_grad()
            loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
            loss.backward()
            layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        with torch.no_grad():
            self.w, self.b = net.get_fc()

            print("Extracting id testing feature")
            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                feature = net(data, return_feature=True)[..., 0, 0].cpu().numpy()
                feature_id_val.append(feature)
            feature_id_val = np.concatenate(feature_id_val, axis=0)
            with open("feature_id_val.pkl", 'wb') as f:
                pickle.dump(feature_id_val, f)

        self.score_id = self.gradnorm(feature_id_val, self.w, self.b)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature_ood = net.forward(data, return_feature = True)[..., 0, 0].cpu()
        score_ood = self.gradnorm(feature_ood, self.w, self.b)
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        return pred, torch.from_numpy(-score_ood)
