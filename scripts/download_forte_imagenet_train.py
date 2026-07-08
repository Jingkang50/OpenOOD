"""Download a random sample of ImageNet-1k train images for the Forte
reference set.

OpenOOD's downloadable `imagenet_1k` package contains only the val split,
but Forte's setup needs a sample of the ID training distribution (no labels
required). This script pulls train shards of the official gated
`ILSVRC/imagenet-1k` HuggingFace dataset (requires an authorized HF token)
and stores the original JPEG bytes.

The train shards are pre-shuffled (each holds ~1000 distinct classes), so a
prefix of shards is a class-balanced random sample.

Note: do NOT substitute mirrors that re-encode images to fixed square sizes
(e.g. 256x256): Forte is a two-sample test, so the reference sample must go
through the same geometric pipeline as the evaluated images -- an
aspect-ratio-destroying resize shifts the reference distribution and
collapses OOD separation.

Usage:
    python scripts/download_forte_imagenet_train.py --n-samples 100000 \
        --out-dir ./data/images_largescale/imagenet_1k_train_subset
"""
import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

TOTAL_SHARDS = 294

parser = argparse.ArgumentParser()
parser.add_argument('--n-samples', type=int, default=100000)
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/images_largescale/imagenet_1k_train_subset')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
existing = len([f for f in os.listdir(args.out_dir) if 'JPEG' in f or
                f.endswith(('.jpg', '.jpeg', '.png'))])
if existing >= args.n_samples:
    print(f'{existing} images already present, nothing to do')
    raise SystemExit

n = 0
progress = tqdm(total=args.n_samples, desc='Downloading')
for shard in range(TOTAL_SHARDS):
    path = hf_hub_download('ILSVRC/imagenet-1k',
                           f'data/train-{shard:05d}-of-{TOTAL_SHARDS:05d}'
                           '.parquet',
                           repo_type='dataset')
    table = pq.read_table(path, columns=['image', 'label'])
    for img, label in zip(table['image'].to_pylist(),
                          table['label'].to_pylist()):
        with open(os.path.join(args.out_dir, f'{n:06d}_{label}.JPEG'),
                  'wb') as f:
            f.write(img['bytes'])
        n += 1
        progress.update(1)
        if n >= args.n_samples:
            break
    os.remove(path)  # drop the hub cache copy; ~450MB per shard
    if n >= args.n_samples:
        break
progress.close()
print(f'Saved {n} images to {args.out_dir}')
