#!/bin/bash
# sh scripts/ood/sem/cifar10_train_sem.sh

# GPU=1
# CPU=1
# node=79
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/train_sem.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 8 \
--network.checkpoint 'results/checkpoints/cifar10_res18_acc94.30.ckpt'
