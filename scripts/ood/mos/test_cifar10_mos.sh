#!/bin/bash
# sh scripts/ood/mos/test_cifar10_mos.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10_double_label.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_mos.yml \
configs/postprocessors/mos.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 1
