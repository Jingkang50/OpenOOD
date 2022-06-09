from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class KLMatchingPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.dim = self.args.dim
        self.num_classes = self.config.dataset.num_classes
        self.net_name = config.network.name

    def kl(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        with torch.no_grad():
            self.w, self.b = net.get_fc()
            print('Extracting id training feature')
            feature_id_train = []
            for batch in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                if self.net_name == 'lenet':
                    _, feature = net.forward_secondary(data,
                                                       return_feature=True)
                else:
                    _, feature = net(data, return_feature=True)
                feature = feature.cpu().numpy()
                feature_id_train.append(feature)
            feature_id_train = np.concatenate(feature_id_train, axis=0)
            logit_id_train = feature_id_train @ self.w.T + self.b
            softmax_id_train = softmax(logit_id_train, axis=-1)
            pred_labels_train = np.argmax(softmax_id_train, axis=-1)
            self.mean_softmax_train = [
                softmax_id_train[pred_labels_train == i].mean(axis=0)
                for i in tqdm(range(self.num_classes))
            ]

            print('Extracting id testing feature')
            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                if self.net_name == 'lenet':
                    _, feature = net.forward_secondary(data,
                                                       return_feature=True)
                else:
                    _, feature = net(data, return_feature=True)
                feature = feature.cpu().numpy()
                feature_id_val.append(feature)
            feature_id_val = np.concatenate(feature_id_val, axis=0)
            logit_id_val = feature_id_val @ self.w.T + self.b
            softmax_id_val = softmax(logit_id_val, axis=-1)
            import pdb
            pdb.set_trace()
            self.score_id = -pairwise_distances_argmin_min(
                softmax_id_val,
                np.array(self.mean_softmax_train),
                metric=self.kl)[1]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.net_name == 'lenet':
            _, feature_ood = net.forward_secondary(data, return_feature=True)
        else:
            _, feature_ood = net(data, return_feature=True)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        softmax_ood = softmax(logit_ood.numpy(), axis=-1)
        _, pred = torch.max(logit_ood, dim=1)
        score_ood = -pairwise_distances_argmin_min(
            softmax_ood, np.array(self.mean_softmax_train), metric=self.kl)[1]
        return pred, torch.from_numpy(score_ood)
