from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from math import ceil
""" Code borrowed from https://github.com/snu-mllab/Neural-Relation-Graph
"""


def normalize(feat, nc=50000):
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) *
                 nc] = feat_ / torch.sqrt((feat_**2).sum(-1) + 1e-10).reshape(
                     -1, 1)

    return feat


def kernel(feat, feat_t, prob, prob_t, split=2):
    """Kernel function (assume feature is normalized)"""
    size = ceil(len(feat_t) / split)
    rel_full = []
    for i in range(split):
        feat_t_ = feat_t[i * size:(i + 1) * size]
        prob_t_ = prob_t[i * size:(i + 1) * size]

        with torch.no_grad():
            dot = torch.matmul(feat, feat_t_.transpose(1, 0))
            dot = torch.clamp(dot, min=0.)

            sim = torch.matmul(prob, prob_t_.transpose(1, 0))
            rel = dot * sim

        rel_full.append(rel)

    rel_full = torch.cat(rel_full, dim=-1)
    return rel_full


def get_relation(feat, feat_t, prob, prob_t, pow=1, chunk=50, thres=0.03):
    """Get relation values (top-k and summation)

    Args:
        feat (torch.Tensor [N,D]): features of the source data
        feat_t (torch.Tensor [N',D]): features of the target data
        prob (torch.Tensor [N,C]): probabilty vectors of the source data
        prob_t (torch.Tensor [N',C]): probabilty vectors of the target data
        pow (int): Temperature of kernel function
        chunk (int): batch size of kernel calculation (trade off between memory and speed)
        thres (float): cut off value for small relation graph edges. Defaults to 0.03.

    Returns:
        graph: statistics of relation graph
    """

    n = feat.shape[0]
    n_chunk = ceil(n / chunk)

    score = []
    for i in range(n_chunk):
        feat_ = feat[i * chunk:(i + 1) * chunk]
        prob_ = prob[i * chunk:(i + 1) * chunk]

        rel = kernel(feat_, feat_t, prob_, prob_t)

        mask = (rel.abs() > thres)
        rel_mask = mask * rel
        edge_sum = (rel_mask.sign() * (rel_mask.abs()**pow)).sum(-1)

        score.append(edge_sum.cpu())

    score = torch.cat(score, dim=0)

    return score


class RelationPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(RelationPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.pow = self.args.pow
        self.feature_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            feature_log = []
            prob_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    logit, feature = net(data, return_feature=True)
                    prob = torch.softmax(logit, dim=1)
                    feature_log.append(normalize(feature))
                    prob_log.append(prob)

            self.feat_train = torch.cat(feature_log, axis=0)
            self.prob_train = torch.cat(prob_log, axis=0)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature = normalize(feature)
        prob = torch.softmax(output, dim=1)

        score = get_relation(feature,
                             self.feat_train,
                             prob,
                             self.prob_train,
                             pow=self.pow)

        _, pred = torch.max(prob, dim=1)

        return pred, score

    def set_hyperparam(self, hyperparam: list):
        self.pow = hyperparam[0]

    def get_hyperparam(self):
        return self.pow
