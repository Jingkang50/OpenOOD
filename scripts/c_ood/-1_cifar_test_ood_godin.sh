#!/bin/bash
# sh scripts/c_ood/-1_cifar_test_godin.sh

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
configs/datasets/objects/cifar10_ood.yml \
configs/networks/godinnet.yml \
configs/pipelines/test/test_ood.yml \
configs/postprocessors/godin.yml \
--dataset.image_size 32 \
--num_workers 8 \
--network.checkpoint ./results/cifar10_deconfnet_generalized_odin_e300_lr0.1/best.ckpt \
--force_merge False
