# python scripts/ood/sem/sweep_osr.py
import os

config = [
    # ['osr_cifar6/cifar6_seed1.yml', 'osr_cifar6/cifar6_seed1_ood.yml', 'resnet18_32x32', 'results/checkpoints/osr/cifar6_seed1.ckpt'],
    [
        'osr_cifar50/cifar50_seed1.yml', 'osr_cifar50/cifar50_seed1_ood.yml',
        'resnet18_32x32', 'results/checkpoints/osr/cifar50_seed1.ckpt'
    ],
    # ['osr_tin20/tin20_seed1.yml', 'osr_tin20/tin20_seed1_ood.yml', 'resnet18_64x64', 'results/checkpoints/osr/tin20_seed1.ckpt'],
    # ['osr_mnist6/mnist6_seed1.yml', 'osr_mnist6/mnist6_seed1_ood.yml', 'lenet', 'results/checkpoints/osr/mnist6_seed1.ckpt'],
]

for [dataset, ood_dataset, network, pth] in config:
    command = (f"PYTHONPATH='.':$PYTHONPATH \
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
    --cpus-per-task=1 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --job-name=openood \
    python main.py \
    --config configs/datasets/{dataset} \
    configs/datasets/{ood_dataset} \
    configs/networks/{network}.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gmm.yml \
    --network.pretrained True \
    --network.checkpoint {pth} \
    --num_workers 8 \
    --merge_option merge &")
    os.system(command)
