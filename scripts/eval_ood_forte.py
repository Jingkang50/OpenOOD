"""Evaluate the Forte OOD detector (Ganguly et al., ICLR 2025) on the
OpenOOD benchmarks.

Forte is training-free and label-free: it scores typicality of test samples
against a reference sample of ID data in frozen self-supervised
representation spaces (CLIP, ViT-MSN, DINOv2). There is no classifier
checkpoint to load, so this script does not take a --root of trained seeds;
the backbones themselves are deterministic and the only stochasticity is the
reference/fit split, controlled by --seed.

Example:
    python scripts/eval_ood_forte.py --id-data cifar10 --batch-size 256
"""
import argparse
import os
import sys

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openood.evaluation_api import Evaluator  # noqa: E402
from openood.evaluation_api.postprocessor import \
    get_postprocessor as get_api_postprocessor  # noqa: E402
from openood.networks.forte_net import ForteNet  # noqa: E402


class FolderDataset(Dataset):
    """Images from a flat folder as OpenOOD-style {'data', 'label'} dicts."""
    def __init__(self, root, transform):
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
parser.add_argument('--backbones',
                    nargs='+',
                    default=['clip', 'vitmsn', 'dinov2'])
parser.add_argument('--variant',
                    type=str,
                    default=None,
                    choices=[None, 'gmm', 'kde', 'ocsvm'],
                    help='override the density estimator in forte.yml')
parser.add_argument('--n-setup-samples', type=int, default=None,
                    help='override the ID-train subsample cap in forte.yml')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=None,
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
    help='folder of ID train images to fit the Forte reference sample on, '
    'for benchmarks whose OpenOOD id train images are not downloadable '
    '(e.g. imagenet); see scripts/download_forte_imagenet_train.py')
args = parser.parse_args()

net = ForteNet(backbones=args.backbones).cuda().eval()

postprocessor = get_api_postprocessor(args.config_root, 'forte', args.id_data)
if args.variant is not None:
    postprocessor.variant = args.variant
if args.n_setup_samples is not None:
    postprocessor.n_setup_samples = args.n_setup_samples
if args.seed is not None:
    postprocessor.seed = args.seed
if args.cache_dir is not None:
    postprocessor.feature_cache_dir = args.cache_dir

if args.setup_dir is not None:
    # pre-fit on a custom ID train sample; the setup_flag guard turns the
    # Evaluator's own setup call into a no-op
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

# ForteNet is not a classifier, so ID accuracy is undefined; skip the
# classification pass that eval_ood would otherwise trigger.
evaluator.metrics['id_acc'] = 0.0
evaluator.metrics['csid_acc'] = 0.0

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
