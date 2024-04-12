#!/bin/bash
# sh scripts/osr/openmax/cifar10_test_ood_openmax.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/tiny_inet/tiny_inat.yml \
    configs/datasets/tiny_inet/tiny_inat_ood.yml \
    configs/networks/convnext_base.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/openmax.yml \
    --num_workers 1 \
    --network.checkpoint 'pretrained_models/checkpoint-best.pth' \
    --mark 0

