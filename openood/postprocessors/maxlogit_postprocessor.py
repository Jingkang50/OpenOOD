from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class MaxLogitPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.w, self.b = net.get_fc()
        net.eval()

        with torch.no_grad():

            print('Extracting id testing feature')
            feature_id_val = []
            for batch in tqdm(id_loader_dict['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                feature = net(data, return_feature=True).cpu().numpy()
                feature_id_val.append(feature)
            feature_id_val = np.concatenate(feature_id_val, axis=0)
        logit_id_val = feature_id_val @ self.w.T + self.b
        self.score_id = logit_id_val.max(axis=-1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature_ood = net.forward(data, return_feature = True).cpu()
        logit_ood = feature_ood.numpy() @ self.w.T + self.b
        score_ood, pred = torch.max(torch.from_numpy(logit_ood), dim=1)
        return pred, score_ood
