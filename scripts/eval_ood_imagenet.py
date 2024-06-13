import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.evaluation_api import Evaluator

from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.cider_net import CIDERNet
from openood.networks.t2fnorm_net import T2FNormNet


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--arch',
                    default='resnet50',
                    choices=['resnet50', 'swin-t', 'vit-b-16', 'regnet'])
parser.add_argument('--tvs-version', default=1, choices=[1, 2])
parser.add_argument('--ckpt-path', default=None)
parser.add_argument('--tvs-pretrained', action='store_true')
parser.add_argument('--postprocessor', default='msp')
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--batch-size', default=200, type=int)
args = parser.parse_args()

if not args.tvs_pretrained:
    assert args.ckpt_path is not None
    root = '/'.join(args.ckpt_path.split('/')[:-1])
else:
    root = os.path.join(
        ROOT_DIR, 'results',
        f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
    if not os.path.exists(root):
        os.makedirs(root)

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor
# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl')):
    with open(
            os.path.join(root, 'postprocessors', f'{postprocessor_name}.pkl'),
            'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

# assuming the model is either
# 1) torchvision pre-trained; or
# 2) a specified checkpoint
if args.tvs_pretrained:
    if args.arch == 'resnet50':
        net = ResNet50()
        weights = eval(f'ResNet50_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'swin-t':
        net = Swin_T()
        weights = eval(f'Swin_T_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'vit-b-16':
        net = ViT_B_16()
        weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'regnet':
        net = RegNet_Y_16GF()
        weights = eval(
            f'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    else:
        raise NotImplementedError
else:
    if args.arch == 'resnet50':
        if postprocessor_name == 'conf_branch':
            net = ConfBranchNet(backbone=ResNet50(), num_classes=1000)
        elif postprocessor_name == 'godin':
            backbone = ResNet50()
            net = GodinNet(backbone=backbone,
                           feature_size=backbone.feature_size,
                           num_classes=1000)
        elif postprocessor_name == 'rotpred':
            net = RotNet(backbone=ResNet50(), num_classes=1000)
        elif postprocessor_name in ['cider', 'reweightood']:
            net = CIDERNet(backbone=ResNet50(),
                           head='mlp',
                           feat_dim=128,
                           num_classes=1000)
        elif postprocessor_name == 't2fnorm':
            net = T2FNormNet(backbone=ResNet50(), num_classes=1000)
        else:
            net = ResNet50()

        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        net.load_state_dict(ckpt)
        preprocessor = trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError

net.cuda()
net.eval()
# a unified evaluator
evaluator = Evaluator(
    net,
    id_name='imagenet',  # the target ID dataset
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=preprocessor,  # default preprocessing
    postprocessor_name=postprocessor_name,
    postprocessor=postprocessor,
    batch_size=args.
    batch_size,  # for certain methods the results can be slightly affected by batch size
    shuffle=True,
    num_workers=8)

# load pre-computed scores if exists
if os.path.isfile(os.path.join(root, 'scores', f'{postprocessor_name}.pkl')):
    with open(os.path.join(root, 'scores', f'{postprocessor_name}.pkl'),
              'rb') as f:
        scores = pickle.load(f)
    update(evaluator.scores, scores)
    print('Loaded pre-computed scores from file.')

# save postprocessor for future reuse
if hasattr(evaluator.postprocessor, 'setup_flag'
           ) or evaluator.postprocessor.hyperparam_search_done is True:
    pp_save_root = os.path.join(root, 'postprocessors')
    if not os.path.exists(pp_save_root):
        os.makedirs(pp_save_root)

    if not os.path.isfile(
            os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
        with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.postprocessor, f, pickle.HIGHEST_PROTOCOL)

# the metrics is a dataframe
metrics = evaluator.eval_ood(fsood=args.fsood)

# saving and recording
if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    if not os.path.isfile(
            os.path.join(saving_root, f'{postprocessor_name}.csv')):
        metrics.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'),
                       float_format='{:.2f}'.format)

if args.save_score:
    score_save_root = os.path.join(root, 'scores')
    if not os.path.exists(score_save_root):
        os.makedirs(score_save_root)
    with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
              'wb') as f:
        pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)
