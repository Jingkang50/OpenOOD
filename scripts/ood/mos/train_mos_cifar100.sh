#!/bin/bash
# sh scripts/ood/mos/train_mos_cifar100.sh

# GPU=1
# CPU=0


# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \

CUDA_VISIBLE_DEVICES=1 python main.py \
--config configs/datasets/cifar100/cifar100_double_label.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/train_mos.yml \
configs/postprocessors/mos.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 1
