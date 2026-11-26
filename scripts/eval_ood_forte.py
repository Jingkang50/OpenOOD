"""Evaluate the Forte OOD detector (Ganguly et al., ICLR 2025) on the
OpenOOD benchmarks.

Forte is training-free and label-free: it scores typicality of test samples
against a reference sample of ID data in frozen self-supervised
representation spaces (CLIP, ViT-MSN, DINOv2). There is no classifier
checkpoint to load, so this script does not take a --root of trained seeds;
the backbones themselves are deterministic and the only stochasticity is the
reference/fit split, controlled by --seed.

Forte is a two-sample test and follows the reference implementation's
protocol exactly (github.com/DebarghaG/forte, main.py): the ID evaluation
pool is split 67/33 into reference+fit and held-out parts, and for every
OOD dataset the held-out ID sample and the OOD set are fused into one
evaluated sample before computing per-point PRDC (`fused_two_sample`), so
ID confidences are pair-specific. No ID train images are required in this
default mode. Passing --setup-dir switches the reference to an external ID
train sample (OpenOOD-style setup; the full ID test set is then evaluated,
and train/test distribution shift lowers the scores).

The encoders default to the paper's (CLIP ViT-B/32, ViT-MSN-B, DINOv2-B)
and can be changed with `backbones` in forte.yml or --backbones. Results
with other encoders are not the paper's method and should be reported with
the encoder set named.

Example:
    python scripts/eval_ood_forte.py --id-data cifar10 --batch-size 256
"""
import argparse
import os
import sys

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openood.evaluation_api import Evaluator  # noqa: E402
from openood.evaluation_api.postprocessor import \
    get_postprocessor as get_api_postprocessor  # noqa: E402
from openood.networks.forte_net import (  # noqa: E402
    DEFAULT_FORTE_BACKBONES, ForteNet)


class FolderDataset(Dataset):
    """Images from a flat folder as OpenOOD-style {'data', 'label'} dicts."""
    def __init__(self, root, transform):
        # identifies this dataset in the postprocessor's feature cache
        self.name = os.path.basename(os.path.normpath(root))
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {
            'data': self.transform(Image.open(self.paths[i])),
            'label': 0,
        }


parser = argparse.ArgumentParser()
parser.add_argument('--id-data',
                    type=str,
                    default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet200', 'imagenet'])
parser.add_argument('--data-root', type=str, default='./data')
parser.add_argument('--config-root', type=str, default='./configs')
parser.add_argument(
    '--backbones',
    nargs='+',
    default=None,
    help='override the encoders in forte.yml: names from '
    'openood.networks.forte_net.FORTE_BACKBONES or Hugging Face hub ids, '
    'optionally with a pooling suffix (e.g. facebook/dinov2-large:cls)')
parser.add_argument('--variant',
                    type=str,
                    default=None,
                    choices=[None, 'gmm', 'kde', 'ocsvm'],
                    help='override the density estimator in forte.yml')
parser.add_argument('--n-setup-samples',
                    type=int,
                    default=None,
                    help='override the ID-train subsample cap in forte.yml')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed',
                    type=int,
                    default=None,
                    help='override the reference/fit split seed in forte.yml')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--save-csv', type=str, default=None)
parser.add_argument(
    '--cache-dir',
    type=str,
    default=None,
    help='directory for caching extracted backbone features across runs')
parser.add_argument(
    '--setup-dir',
    type=str,
    default=None,
    help='optional folder of ID train images for an external (OpenOOD-'
    'style) reference sample instead of the default paper protocol, which '
    'splits the ID test pool; see scripts/download_forte_imagenet_train.py')
args = parser.parse_args()

postprocessor = get_api_postprocessor(args.config_root, 'forte', args.id_data)
if args.backbones is None:
    args.backbones = list(
        getattr(postprocessor.args, 'backbones', DEFAULT_FORTE_BACKBONES))
net = ForteNet(backbones=args.backbones).cuda().eval()

if args.variant is not None:
    postprocessor.variant = args.variant
if args.n_setup_samples is not None:
    postprocessor.n_setup_samples = args.n_setup_samples
if args.seed is not None:
    postprocessor.seed = args.seed
if args.cache_dir is not None:
    postprocessor.feature_cache_dir = args.cache_dir

if args.setup_dir is not None:
    # pre-fit on an external ID train sample; the setup_flag guard turns
    # the Evaluator's own setup call into a no-op
    postprocessor.reference_pool = 'train'
    setup_loader = DataLoader(FolderDataset(args.setup_dir,
                                            ForteNet.get_preprocessor()),
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers)
    postprocessor.setup(net, {'train': setup_loader}, {})

evaluator = Evaluator(
    net,
    id_name=args.id_data,
    data_root=args.data_root,
    config_root=args.config_root,
    preprocessor=ForteNet.get_preprocessor(),
    postprocessor=postprocessor,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

# ForteNet is not a classifier, so ID accuracy is undefined; report NaN
# and skip the classification pass that eval_ood would otherwise trigger.
evaluator.metrics['id_acc'] = float('nan')
evaluator.metrics['csid_acc'] = float('nan')

metrics = evaluator.eval_ood(fsood=args.fsood)

if args.save_csv:
    os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
    metrics.to_csv(args.save_csv)
    print(f'Saved metrics to {args.save_csv}')

with pd.option_context('display.float_format', '{:,.2f}'.format):
    print('\nFinal results '
          f'(forte-{postprocessor.variant}, k={postprocessor.nearest_k}, '
          f'backbones={args.backbones}):')
    print(metrics)
