#!/bin/bash
# sh scripts/uncertainty/cutout/cifar100_train_cutout.sh

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/cutout_preprocessor.yml \
    --preprocessor.length 8 \
    --seed 0 \
    --mark cutout-1-8
