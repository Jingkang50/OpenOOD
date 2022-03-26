from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def compute_threshold(net: nn.Module, data_loader: DataLoader):

    activation_log = []
    net.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Eval: ', position=0, leave=True):
            data = batch['data'].cuda()
            data = data.float()

            batch_size = data.shape[0]
            layer_key = 'avgpool'
            net.avgpool.register_forward_hook(get_activation(layer_key))

            net(data)

            feature = activation[layer_key]
            dim = feature.shape[1]
            activation_log.append(feature.data.cpu().numpy().reshape(
                batch_size, dim, -1).mean(2))

    activation_log = np.concatenate(activation_log, axis=0)
    threshold = np.percentile(activation_log.flatten(), 90)
    print('THRESHOLD at percentile is: {}'.format(threshold))

    return threshold


class ReactEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(ReactEvaluator, self).__init__(config)

    def eval_ood(self, net: nn.Module, id_data_loader: DataLoader,
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        threshold = compute_threshold(net, id_data_loader['test'])
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(id_data_loader['test'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()
                target = batch['label'].cuda()

                output = net.forward_threshold(data, threshold)
                loss = F.cross_entropy(output, target)

                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                loss_avg += float(loss.data)
        acc = correct / len(id_data_loader['test'].dataset)
        print(acc)

    def compute_threshold(self, net: nn.Module, data_loader: DataLoader):

        activation_log = []
        net.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()

                batch_size = data.shape[0]
                layer_key = 'avgpool'
                net.avgpool.register_forward_hook(get_activation(layer_key))

                net(data)

                feature = activation[layer_key]
                dim = feature.shape[1]
                activation_log.append(feature.data.cpu().numpy().reshape(
                    batch_size, dim, -1).mean(2))

        activation_log = np.concatenate(activation_log, axis=0)
        threshold = np.percentile(activation_log.flatten(), 90)
        print('THRESHOLD at percentile is: {}'.format(threshold))

        return threshold
