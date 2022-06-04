from typing import Any

import numpy as np
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor
import faiss

from tqdm import tqdm


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class DICEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DICEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.mean_act = None
        self.masked_w = None

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
                net.avgpool.register_forward_hook(
                    get_activation(layer_key))

                net(data)

                feature = activation[layer_key]
                dim = feature.shape[1]
                activation_log.append(feature.data.cpu().numpy().reshape(
                    batch_size, dim, -1).mean(2))

        activation_log = np.concatenate(activation_log, axis=0)
        self.mean_act = activation_log.mean(0)

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.args.p)
        mask = torch.Tensor((contrib > self.thresh)).cuda()
        self.masked_w = w * mask

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.masked_w is None:
            self.calculate_mask(net.fc.weight)
        _, feature = net(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + net.fc.bias
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf
