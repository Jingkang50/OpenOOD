#!/bin/bash
# sh scripts/ad/patchcore/cifar10_test_ood_patchcore.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/patchcore_net.yml \
configs/pipelines/test/test_patchcore.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/patch.yml \
--num_workers 8 \
--merge_option merge
