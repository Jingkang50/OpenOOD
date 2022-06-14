# python scripts/uncertainty/pixmix/sweep.py
import os

config = [
    ['osr_cifar6/cifar6_seed1.yml', 'resnet18_32x32', 'cifar10'],
    ['osr_cifar50/cifar50_seed1.yml', 'resnet18_32x32', 'cifar100'],
    ['osr_tin20/tin20_seed1.yml', 'resnet18_64x64', 'tin'],
]

for [dataset, network, od] in config:
    command = (f"PYTHONPATH='.':$PYTHONPATH \
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
    --cpus-per-task=1 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --job-name=openood \
    python main.py \
    --config configs/datasets/{dataset} \
    configs/networks/{network}.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/pixmix_preprocessor.yml \
    --optimizer.num_epochs 100 \
    --dataset.name {od}_osr \
    --num_workers 8")
    os.system(command)
