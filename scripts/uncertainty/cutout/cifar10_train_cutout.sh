#!/bin/bash
# sh scripts/uncertainty/cutout/cifar10_train_cutout.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/cutout_preprocessor.yml \
    --seed 0 \
    --mark cutout-1-16
