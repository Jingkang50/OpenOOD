#!/bin/bash
# sh scripts/ood/mos/test_mos.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar100/cifar100_double_label.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/mos.yml \
configs/pipelines/test/test_mos.yml \
configs/postprocessors/mos.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 1
