#!/bin/bash
# sh scripts/uncertainty/temp_scaling/cifar10_test_ood_tempscaling.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/temp_scaling.yml \
--num_workers 8 \
--network.checkpoint 'results/_cifar10_resnet18_32x32_base_e100_lr0.1/best_epoch89_acc0.9430.ckpt' \
--mark 0
