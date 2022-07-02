#!/bin/bash
# sh scripts/ad/patchcore/patchcore.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/patchcorenet.yml \
configs/pipelines/test/test_patchcore.yml \
configs/postprocessors/patch.yml \
configs/preprocessors/base_preprocessor.yml \
--evaluator.name ad \
--num_workers 8 \
--merge_option merge
