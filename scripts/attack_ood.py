import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import argparse
import pickle
import collections
from glob import glob

import torch

from openood.evaluation_api import AttackDataset

from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet

from openood.attacks.misc import (
    convert_to_float,
    str2bool
)

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200'])
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument("--att",  default="pgd", choices=[None, 'fgsm', 'bim', 'pgd', 'df', 'cw', 'mpgd'], help="")
parser.add_argument("--eps",  default="4/255", help="")
parser.add_argument("--norm",  default="Linf", choices=['Linf', 'L2', 'L1'], help="")
parser.add_argument('--masked-patch-size', default=60, type=int)
parser.add_argument('--take-model', type=int, default=-1)

parser.add_argument("--debug",  default=False, type=str2bool, help="")
args = parser.parse_args()

root = args.root
if args.eps and "." in args.eps:
    eps = float(args.eps)
else:
    eps = args.eps.replace("/", "") if args.eps and "/" in args.eps else ""; print("eps", eps)
args.eps = convert_to_float(args.eps)

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}

model_architecture = {
    'cifar10': "ResNet18_32x32",
    'cifar100': "ResNet18_32x32",
    'imagenet200': "ResNet18_224x224",
}

args.arch = model_architecture[args.id_data]

try:
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# assume that the root folder contains subfolders each corresponding to
# a training run, e.g., s0, s1, s2
# this structure is automatically created if you use OpenOOD for train
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# iterate through training runs
all_metrics = []
subfolders = glob(os.path.join(root, 's*'))
# todo expand to variation
subfolders = sorted(glob(os.path.join(root, 's2')))
subfolder = subfolders[args.take_model]

# load pre-setup postprocessor if exists
if os.path.isfile(
        os.path.join(subfolder, 'postprocessors',
                        f'{postprocessor_name}.pkl')):
    with open(
            os.path.join(subfolder, 'postprocessors',
                            f'{postprocessor_name}.pkl'), 'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

# load the pretrained model provided by the user
if postprocessor_name == 'conf_branch':
    net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                        num_classes=num_classes)
elif postprocessor_name == 'godin':
    backbone = model_arch(num_classes=num_classes)
    net = GodinNet(backbone=backbone,
                    feature_size=backbone.feature_size,
                    num_classes=num_classes)
elif postprocessor_name == 'rotpred':
    net = RotNet(backbone=model_arch(num_classes=num_classes),
                    num_classes=num_classes)
elif 'csi' in root:
    backbone = model_arch(num_classes=num_classes)
    net = CSINet(backbone=backbone,
                    feature_size=backbone.feature_size,
                    num_classes=num_classes)
elif 'udg' in root:
    backbone = model_arch(num_classes=num_classes)
    net = UDGNet(backbone=backbone,
                    num_classes=num_classes,
                    num_clusters=1000)
elif postprocessor_name == 'cider':
    backbone = model_arch(num_classes=num_classes)
    net = CIDERNet(backbone,
                    head='mlp',
                    feat_dim=128,
                    num_classes=num_classes)
elif postprocessor_name == 'npos':
    backbone = model_arch(num_classes=num_classes)
    net = NPOSNet(backbone,
                    head='mlp',
                    feat_dim=128,
                    num_classes=num_classes)
else:
    net = model_arch(num_classes=num_classes)

net.load_state_dict(
    torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
net.cuda()
net.eval()


attackdataset = AttackDataset(
net,
id_name=args.id_data,  # the target ID dataset
data_root=os.path.join(ROOT_DIR, 'data'),
config_root=os.path.join(ROOT_DIR, 'configs'),
preprocessor=None,  # default preprocessing
batch_size=args.
batch_size,  # for certain methods the results can be slightly affected by batch size
shuffle=False,
num_workers=8)

attackdataset.run_attack(args)
