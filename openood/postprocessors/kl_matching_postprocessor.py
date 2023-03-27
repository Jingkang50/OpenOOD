from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class KLMatchingPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.dim = self.args.dim
        self.num_classes = self.config.dataset.num_classes
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def kl(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

        print('Extracting id validation softmax posterior distributions')
        all_softmax = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['val'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                logits = net(data)
                all_softmax.append(F.softmax(logits, 1).cpu())
                preds.append(logits.argmax(1).cpu())

        all_softmax = torch.cat(all_softmax)
        preds = torch.cat(preds)
        self.mean_softmax_val = [
            all_softmax[preds.eq(i)].mean(0).numpy()
            for i in tqdm(range(self.num_classes))
        ]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
