#!/bin/bash
# sh scripts/ood/csi/cifar100_train_csi_step1.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/csi_net.yml \
    configs/pipelines/train/train_csi.yml \
    configs/preprocessors/csi_preprocessor.yml \
    --network.pretrained False \
    --optimizer.num_epochs 100 \
    --dataset.train.batch_size 64 \
    --merge_option merge \
    --mode csi_step1 \
    --seed 0
