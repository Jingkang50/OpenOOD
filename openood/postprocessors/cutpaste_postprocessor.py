from __future__ import division, print_function

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf as LW
from torch.utils.data import DataLoader
from tqdm import tqdm


class CutPastePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # get train embeds
        train_loader = id_loader_dict['train']
        train_embed = []
        train_dataiter = iter(train_loader)
        with torch.no_grad():
            for train_step in tqdm(range(1,
                                         len(train_dataiter) + 1),
                                   desc='Train embeds'):
                batch = next(train_dataiter)
                data = torch.cat(batch['data'], 0)
                if (np.array(data).shape[0] == 4):
                    data = data.numpy().tolist()
                    data = data[0:len(data) // 2]
                    data = torch.Tensor(data)
                data = data.cuda()
                embed, logit = net(data)
                train_embed.append(embed.cuda())
        train_embeds = torch.cat(train_embed)
        self.train_embeds = torch.nn.functional.normalize(train_embeds,
                                                          p=2,
                                                          dim=1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # get embeds
        embeds = []
        embed, output = net(data)
        embeds.append(embed.cuda())
        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        # compute distances
        density = GaussianDensityTorch()
        density.fit(self.train_embeds)
        distances = density.predict(embeds)
        distances = 200 - distances
        return pred, distances

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = torch.cat(batch['data'], 0)
            data = data.cuda()
            # label = torch.arange(2)
            label = torch.tensor([0, -1])
            label = label.repeat_interleave(len(batch['data'][0])).cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list


class Density(object):
    def fit(self, embeddings):
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError


class GaussianDensityTorch(Density):
    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, axis=0)
        self.inv_cov = torch.Tensor(LW().fit(embeddings.cpu()).precision_,
                                    device='cpu')

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean,
                                              self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(values: torch.Tensor, mean: torch.Tensor,
                             inv_covariance: torch.Tensor) -> torch.Tensor:

        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        inv_covariance = inv_covariance.cuda()
        dist = torch.einsum('im,mn,in->i', x_mu, inv_covariance, x_mu)

        return dist.sqrt()
