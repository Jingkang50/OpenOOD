#!/bin/bash
# sh scripts/basics/imagenet200/train_imagenet200.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    --seed 0
