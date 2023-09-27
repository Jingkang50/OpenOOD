from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.special import logsumexp
from copy import deepcopy
from .base_postprocessor import BasePostprocessor
from torch.utils.data import DataLoader

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def knn_score(bankfeas, queryfeas, k=100, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))


    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, I = index.search(queryfeas, k)

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores

class NNGuidePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NNGuidePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            ## NNGuide
            bank_feas = []
            bank_logits = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                desc='Setup: ',
                                position=0,
                                leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    logit, feature = net(data, return_feature=True)
                    bank_feas.append(
                        normalizer(feature.data.cpu().numpy()))
                    bank_logits.append(logit.data.cpu().numpy())
                    if len(bank_feas) * id_loader_dict['train'].batch_size > int(len(id_loader_dict['train'].dataset) * self.alpha):
                        break

            bank_feas = np.concatenate(bank_feas, axis=0)
            bank_confs = logsumexp(np.concatenate(bank_logits, axis=0), axis=-1)
            self.bank_guide = bank_feas * bank_confs[:, None]

            self.setup_flag = True
        else:
            pass


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logit, feature = net(data, return_feature=True)
        feas_norm = normalizer(feature.data.cpu().numpy())
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)
        score = conf * energy

        _, pred = torch.max(torch.softmax(logit, dim=1), dim=1)
        return pred, torch.from_numpy(score)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]
        self.percentile = hyperparam[1]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))


    def get_hyperparam(self):
        return [self.K, self.percentile]
