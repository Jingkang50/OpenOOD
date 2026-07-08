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
# count only files this script writes ({index:06d}_{label}.JPEG), so foreign
# images in out-dir can never satisfy the target
existing = len([f for f in os.listdir(args.out_dir) if f.endswith('.JPEG')])
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
    parquet = pq.ParquetFile(path)
    # stream row groups so only one batch is in memory and the final shard
    # stops parsing as soon as the target count is reached
    for batch in parquet.iter_batches(columns=['image', 'label']):
        for img, label in zip(
                batch.column('image').to_pylist(),
                batch.column('label').to_pylist()):
            out_path = os.path.join(args.out_dir, f'{n:06d}_{label}.JPEG')
            # resume: files written by an interrupted run are kept
            if not os.path.exists(out_path):
                with open(out_path, 'wb') as f:
                    f.write(img['bytes'])
            n += 1
            progress.update(1)
            if n >= args.n_samples:
                break
        if n >= args.n_samples:
            break
    parquet.close()
    # drop the hub cache copy (~450MB per shard): hf_hub_download returns a
    # symlink into the cache's blobs/ store, so remove the blob it points
    # at, then the symlink itself
    blob = os.path.realpath(path)
    os.remove(path)
    if blob != path and os.path.exists(blob):
        os.remove(blob)
    if n >= args.n_samples:
        break
progress.close()
print(f'Saved {n} images to {args.out_dir}')
