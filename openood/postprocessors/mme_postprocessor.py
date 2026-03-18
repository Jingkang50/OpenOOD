from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from copy import deepcopy
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .vim_postprocessor import VIMPostprocessor
from .fdbd_postprocessor import fDBDPostprocessor
from .scale_postprocessor import ScalePostprocessor
from .vra_postprocessor import VRAPostprocessor
from .pca_nre_postprocessor import PCANREPostprocessor

from openood.networks.scale_net import ScaleNet
from openood.utils.config import Config


class MMEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.temperature = self.args.temperature
        self.lambd = self.args.lambd
        self.EPSILON = 1e-8

        self.vim = VIMPostprocessor(Config('configs/postprocessors/vim.yml'))
        self.fdbd = fDBDPostprocessor(
            Config('configs/postprocessors/fdbd.yml'))
        self.scale = ScalePostprocessor(
            Config('configs/postprocessors/scale.yml'))
        self.vra = VRAPostprocessor(Config('configs/postprocessors/vra.yml'))
        self.pca_nre = PCANREPostprocessor(
            Config('configs/postprocessors/pca_nre.yml'))
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            self.vim.setup(net, id_loader_dict, ood_loader_dict)
            self.fdbd.setup(net, id_loader_dict, ood_loader_dict)
            self.fdbd.denominator_matrix = self.fdbd.denominator_matrix.cpu()
            self.fdbd.train_mean = self.fdbd.train_mean.cpu()
            self.scale.setup(net, id_loader_dict, ood_loader_dict)
            self.scale.net = ScaleNet(net)
            self.vra.setup(net, id_loader_dict, ood_loader_dict)
            self.pca_nre.setup(net, id_loader_dict, ood_loader_dict)
            net.eval()

            all_activation_log = []
            all_labels = []
            with torch.no_grad():
                self.w, self.b = net.get_fc()
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Eval: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label']
                    all_labels.append(deepcopy(labels))

                    logits, features = net(data, return_feature=True)
                    all_activation_log.append(features.cpu())

            all_labels = torch.cat(all_labels)
            all_activation_log = torch.cat(all_activation_log)

            num_classes = int(all_labels.max().item() + 1)
            feature_dim = all_activation_log.shape[1]
            self.class_means = torch.zeros((num_classes, feature_dim),
                                           dtype=torch.float)
            for i in tqdm(range(num_classes)):
                vectors = all_activation_log[all_labels == i]
                vectors = vectors / (vectors.norm(dim=1, keepdim=True) +
                                     self.EPSILON)
                mean = vectors.mean(dim=0)
                mean = mean / (mean.norm() + self.EPSILON)
                self.class_means[i, :] = mean
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logit_scale_ood = self.scale.net.forward_threshold(
            data, self.scale.percentile)
        _, feature_ood = net.forward(data, return_feature=True)
        feature_vra_ood = feature_ood.clip(min=self.vra.threshold_low,
                                           max=self.vra.threshold_high)

        feature_ood = feature_ood.cpu()
        feature_vra_ood = feature_vra_ood.cpu()

        vlogit_ood = norm(np.matmul(feature_vra_ood.numpy() - self.vim.u,
                                    self.vim.NS),
                          axis=-1) * self.vim.alpha

        rec_norm = np.linalg.norm(
            (feature_vra_ood - self.pca_nre.activation_log_mean)
            @ (np.identity(self.pca_nre.dim) - self.pca_nre.M),
            axis=-1)
        r_ood = rec_norm / np.linalg.norm(feature_vra_ood, axis=-1)

        vectors_ood = (
            feature_ood.T /
            (np.linalg.norm(feature_ood.T, axis=0) + self.EPSILON)).T
        distances_ood = cdist(self.class_means, vectors_ood, 'sqeuclidean')
        distances_ood = distances_ood.T

        logit_vra_ood = feature_vra_ood @ self.w.T + self.b

        values, nn_idx = logit_vra_ood.max(1)
        logits_sub = torch.abs(logit_vra_ood -
                               values.repeat(self.fdbd.num_classes, 1).T)
        if self.fdbd.distance_as_normalizer:
            fdbd_ood = torch.sum(
                logits_sub / self.fdbd.denominator_matrix[nn_idx],
                axis=1) / torch.norm(feature_vra_ood - self.fdbd.train_mean,
                                     dim=1)
        else:
            fdbd_ood = torch.sum(
                logits_sub / self.fdbd.denominator_matrix[nn_idx],
                axis=1) / torch.norm(feature_vra_ood, dim=1)
        fdbd_ood = fdbd_ood.numpy()

        logits_ = torch.logsumexp(logit_scale_ood, dim=-1)
        logits_ = logits_.cpu().numpy()
        max_logits_indices = logit_scale_ood.argmax(dim=1)
        max_logits_indices = max_logits_indices.cpu().numpy()

        distances_ood = self.softmax_temperature(distances_ood,
                                                 axis=1,
                                                 temperature=self.temperature)
        distances_ood = 1 / distances_ood
        distance_ = np.max(distances_ood, axis=1)
        max_distance_indices = np.argmax(distances_ood, axis=1)

        index_matches_bool = (max_logits_indices == max_distance_indices)
        index_matches_float = np.where(index_matches_bool, self.lambd, 1.0)

        score = np.exp(logits_ -
                       vlogit_ood) * distance_ * index_matches_float * (
                           1 - r_ood) * fdbd_ood
        max_float = np.finfo(score.dtype).max
        score[score == np.inf] = max_float
        _, pred = torch.max(logit_vra_ood, dim=1)
        return pred, torch.from_numpy(score)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim

    def softmax_temperature(self, x, axis=None, temperature=1.0):
        x = x / temperature
        x_max = np.amax(x, axis=axis, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
