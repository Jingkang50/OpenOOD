#!/bin/bash
# sh scripts/osr/openmax/cifar10_test_ood_openmax.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/openmax.yml \
--num_workers 8 \
--network.checkpoint 'results/_cifar10_resnet18_32x32_base_e100_lr0.1/best_epoch89_acc0.9430.ckpt' \
--mark 0
