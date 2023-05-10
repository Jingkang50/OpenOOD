from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NPOSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NPOSPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()

                    feature = net.intermediate_forward(data)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature = net.intermediate_forward(data)
        D, _ = self.index.search(
            feature.cpu().numpy(),  # feature is already normalized within net
            self.K,
        )
        kth_dist = -D[:, -1]
        # put dummy prediction here
        # as cider only trains the feature extractor
        pred = torch.zeros(len(kth_dist))
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
