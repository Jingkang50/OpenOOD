from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.activation_log = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        activation_log = []
        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()

                batch_size = data.shape[0]
                layer_key = 'avgpool'
                # net.avgpool.register_forward_hook(get_activation(layer_key))
                net.avgpool.register_forward_hook(get_activation(layer_key))

                net(data)

                feature = activation[layer_key]
                dim = feature.shape[1]
                activation_log.append(
                    normalizer(feature.data.cpu().numpy().reshape(
                        batch_size, dim, -1).mean(2)))

        self.activation_log = np.concatenate(activation_log, axis=0)
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.args.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)
