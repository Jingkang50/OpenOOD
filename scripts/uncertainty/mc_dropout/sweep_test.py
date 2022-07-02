# python scripts/uncertainty/mc_dropout/sweep_test.py
import os

config = [
    [
        'osr_cifar6/cifar6_seed1.yml', 'osr_cifar6/cifar6_seed1_ood.yml',
        'resnet18_32x32',
        'osr_cifar6_seed1_dropout_net_base_e100_lr0.1_default'
    ],
    [
        'osr_cifar50/cifar50_seed1.yml', 'osr_cifar50/cifar50_seed1_ood.yml',
        'resnet18_32x32',
        'osr_cifar50_seed1_dropout_net_base_e100_lr0.1_default'
    ],
    [
        'osr_tin20/tin20_seed1.yml', 'osr_tin20/tin20_seed1_ood.yml',
        'resnet18_64x64', 'osr_tin20_seed1_dropout_net_base_e100_lr0.1_default'
    ],
    [
        'osr_mnist4/mnist4_seed1.yml', 'osr_mnist4/mnist4_seed1_ood.yml',
        'lenet', 'osr_mnist4_seed1_dropout_net_base_e100_lr0.1_default'
    ],
]

for [dataset, ood_data, network, pth] in config:
    command = (f"PYTHONPATH='.':$PYTHONPATH \
    srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 \
    --cpus-per-task=1 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --job-name=openood \
    python main.py \
    --config configs/datasets/{dataset} \
    configs/datasets/{ood_data} \
    configs/networks/dropout_net.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/dropout.yml \
    --num_workers 8 \
    --network.checkpoint 'results/{pth}/best.ckpt' \
    --mark 0 \
    --merge_option merge &")
    os.system(command)
