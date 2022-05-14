#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1 sh scripts/0_basics/cifar10_train_DDT.sh

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
--config configs/datasets/objects/cifar10.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/baseline.yml \
--dataset.image_size 32 \
--optimizer.num_epochs 5 \
--num_workers 4 \
--num_gpus 2 \
--exp_name "'@{dataset.name}'_'@{network.name}'_'@{trainer.name}'_ddt_e'@{optimizer.num_epochs}'_lr'@{optimizer.lr}'" \
--force_merge True
