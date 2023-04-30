from __future__ import division, print_function

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class GRAMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.powers = self.postprocessor_args.powers

        self.feature_min, self.feature_max = None, None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            self.feature_min, self.feature_max = sample_estimator(
                net, id_loader_dict['train'], self.num_classes, self.powers)
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        preds, deviations = get_deviations(net, data, self.feature_min,
                                           self.feature_max, self.num_classes,
                                           self.powers)
        return preds, deviations

    def set_hyperparam(self, hyperparam: list):
        self.powers = hyperparam[0]

    def get_hyperparam(self):
        return self.powers


def tensor2list(x):
    return x.data.cuda().tolist()


@torch.no_grad()
def sample_estimator(model, train_loader, num_classes, powers):

    model.eval()

    num_layer = 5  # 4 for lenet
    num_poles_list = powers
    num_poles = len(num_poles_list)
    feature_class = [[[None for x in range(num_poles)]
                      for y in range(num_layer)] for z in range(num_classes)]
    label_list = []
    mins = [[[None for x in range(num_poles)] for y in range(num_layer)]
            for z in range(num_classes)]
    maxs = [[[None for x in range(num_poles)] for y in range(num_layer)]
            for z in range(num_classes)]

    # collect features and compute gram metrix
    for batch in tqdm(train_loader, desc='Compute min/max'):
        data = batch['data'].cuda()
        label = batch['label']
        _, feature_list = model(data, return_feature_list=True)
        label_list = tensor2list(label)
        for layer_idx in range(num_layer):

            for pole_idx, p in enumerate(num_poles_list):
                temp = feature_list[layer_idx].detach()

                temp = temp**p
                temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                temp = ((torch.matmul(temp,
                                      temp.transpose(dim0=2,
                                                     dim1=1)))).sum(dim=2)
                temp = (temp.sign() * torch.abs(temp)**(1 / p)).reshape(
                    temp.shape[0], -1)

                temp = tensor2list(temp)
                for feature, label in zip(temp, label_list):
                    if isinstance(feature_class[label][layer_idx][pole_idx],
                                  type(None)):
                        feature_class[label][layer_idx][pole_idx] = feature
                    else:
                        feature_class[label][layer_idx][pole_idx].extend(
                            feature)
    # compute mins/maxs
    for label in range(num_classes):
        for layer_idx in range(num_layer):
            for poles_idx in range(num_poles):
                feature = torch.tensor(
                    np.array(feature_class[label][layer_idx][poles_idx]))
                current_min = feature.min(dim=0, keepdim=True)[0]
                current_max = feature.max(dim=0, keepdim=True)[0]

                if mins[label][layer_idx][poles_idx] is None:
                    mins[label][layer_idx][poles_idx] = current_min
                    maxs[label][layer_idx][poles_idx] = current_max
                else:
                    mins[label][layer_idx][poles_idx] = torch.min(
                        current_min, mins[label][layer_idx][poles_idx])
                    maxs[label][layer_idx][poles_idx] = torch.max(
                        current_min, maxs[label][layer_idx][poles_idx])

    return mins, maxs


def get_deviations(model, data, mins, maxs, num_classes, powers):
    model.eval()

    num_layer = 5  # 4 for lenet
    num_poles_list = powers
    exist = 1
    pred_list = []
    dev = [0 for x in range(data.shape[0])]

    # get predictions
    logits, feature_list = model(data, return_feature_list=True)
    confs = F.softmax(logits, dim=1).cpu().detach().numpy()
    preds = np.argmax(confs, axis=1)
    predsList = preds.tolist()
    preds = torch.tensor(preds)

    for pred in predsList:
        exist = 1
        if len(pred_list) == 0:
            pred_list.extend([pred])
        else:
            for pred_now in pred_list:
                if pred_now == pred:
                    exist = 0
            if exist == 1:
                pred_list.extend([pred])

    # compute sample level deviation
    for layer_idx in range(num_layer):
        for pole_idx, p in enumerate(num_poles_list):
            # get gram metirx
            temp = feature_list[layer_idx].detach()
            temp = temp**p
            temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
            temp = ((torch.matmul(temp, temp.transpose(dim0=2,
                                                       dim1=1)))).sum(dim=2)
            temp = (temp.sign() * torch.abs(temp)**(1 / p)).reshape(
                temp.shape[0], -1)
            temp = tensor2list(temp)

            # compute the deviations with train data
            for idx in range(len(temp)):
                dev[idx] += (F.relu(mins[preds[idx]][layer_idx][pole_idx] -
                                    sum(temp[idx])) /
                             torch.abs(mins[preds[idx]][layer_idx][pole_idx] +
                                       10**-6)).sum()
                dev[idx] += (F.relu(
                    sum(temp[idx]) - maxs[preds[idx]][layer_idx][pole_idx]) /
                             torch.abs(maxs[preds[idx]][layer_idx][pole_idx] +
                                       10**-6)).sum()
    conf = [i / 50 for i in dev]

    return preds, torch.tensor(conf)
