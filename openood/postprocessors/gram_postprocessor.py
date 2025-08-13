from __future__ import division, print_function

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
from collections import defaultdict, Counter
import random 

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
        net = FeatureExtractor(net)
        if not self.setup_flag:
            self.feature_min, self.feature_max, self.normalize_factors = sample_estimator(
                net, id_loader_dict['train'], self.num_classes, self.powers)
            self.setup_flag = True
        else:
            pass
        net.destroy_hooks()

    def postprocess(self, net: nn.Module, data: Any):
        net = FeatureExtractor(net)
        preds, deviations = get_deviations(net, data, self.feature_min,
                                           self.feature_max, self.normalize_factors,
                                           self.powers)
        net.destroy_hooks()
        return preds, deviations

    def set_hyperparam(self, hyperparam: list):
        self.powers = hyperparam[0]

    def get_hyperparam(self):
        return self.powers


def tensor2list(x):
    return x.data.cuda().tolist()

def G_p(ob, p):
    temp = ob.detach()
    
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp

def delta(mins, maxs, x):
    dev =  (F.relu(mins-x)/torch.abs(mins+10**-6)).sum(dim=1,keepdim=True)
    dev +=  (F.relu(x-maxs)/torch.abs(maxs+10**-6)).sum(dim=1,keepdim=True)
    return dev

class FeatureExtractor(torch.nn.Module):
    # Inspired from https://github.com/paaatcha/gram-ood
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model
        self.feat_list = list()
        def _hook_fn(_, input, output):
            self.feat_list.append(output)
        
        # To set a different layer, you must use this function:
        def hook_layers(torch_model):
            hooked_layers = list()
            for layer in torch_model.modules():
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Conv2d):
                    hooked_layers.append(layer)
            return hooked_layers

        def register_layers(layers):
            regs_layers = list()
            for lay in layers:
                regs_layers.append(lay.register_forward_hook(_hook_fn))
            return regs_layers
        
        ## Setting the hook
        hl = hook_layers (torch_model)
        self.rgl = register_layers (hl)
        # print(f"{len(self.rgl)} Features")
    
    def forward(self, x, return_feature_list=True):
        preds = self.torch_model(x)
        list = self.feat_list.copy()
        self.feat_list.clear()
        return preds, list
    
    def destroy_hooks(self):
        for lay in self.rgl:
            lay.remove()

@torch.no_grad()
def sample_estimator(model, train_loader, num_classes, powers):

    model.eval()
    gram_features = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda : None)))
    mins = dict()
    maxs = dict()
    class_counts = Counter()
    # collect features and compute gram metrix
    for batch in tqdm(train_loader, desc='Compute min/max'):
        data = batch['data'].cuda()
        label = batch['label']
        _, feature_list = model(data, return_feature_list=True)
        class_counts.update(Counter(label.cpu().numpy()))
        for layer_idx, feature in enumerate(feature_list):
            for power in powers:
                gram_feature = G_p(feature, power).cpu()
                for class_ in range(num_classes):
                    if gram_features[layer_idx][power][class_] is None:
                        gram_features[layer_idx][power][class_] = gram_feature[label==class_]
                    else:
                        gram_features[layer_idx][power][class_] = torch.cat([gram_features[layer_idx][power][class_],gram_feature[label==class_]],dim=0)
    
    val_idxs = {}
    train_idxs = {}
    for c in class_counts:
        L = class_counts[c]
        val_idxs[c] = random.sample(range(L),int(0.1*L))
        train_idxs[c] = list(set(range(L)) - set(val_idxs[c]))
    normalize_factors = []
    # compute mins/maxs
    for layer_idx in gram_features:
        total_delta = None
        for class_ in class_counts:
            trn = train_idxs[class_]
            val = val_idxs[class_]
            class_deltas = 0
            for power in powers:
                mins[layer_idx,power,class_] = gram_features[layer_idx][power][class_][trn].min(dim=0,keepdim=True)[0]
                maxs[layer_idx,power,class_] = gram_features[layer_idx][power][class_][trn].max(dim=0,keepdim=True)[0]
                class_deltas += delta(mins[layer_idx,power,class_],
                                maxs[layer_idx,power,class_],
                                gram_features[layer_idx][power][class_][val])
            if total_delta is None:
                total_delta = class_deltas 
            else:
                total_delta = torch.cat([total_delta,class_deltas],dim=0)
        normalize_factors.append(total_delta.mean(dim=0,keepdim=True))
    normalize_factors = torch.cat(normalize_factors,dim=1)
    return mins, maxs, normalize_factors

def get_deviations(model, data, mins, maxs, normalize_factors, powers):
    model.eval()

    deviations = torch.zeros(data.shape[0],1)

    # get predictions
    logits, feature_list = model(data, return_feature_list=True)
    confs = F.softmax(logits, dim=1).cpu().detach()
    confs, preds = confs.max(dim=1)
    for layer_idx, feature in enumerate(feature_list):
            n = normalize_factors[:,layer_idx].item()
            for power in powers:
                gram_feature = G_p(feature, power).cpu()
                for class_ in range(logits.shape[1]):
                    deviations[preds==class_] += delta(mins[layer_idx,power,class_],
                                maxs[layer_idx,power,class_],
                                gram_feature[preds==class_])/n

    return preds, -deviations/confs[:,None]