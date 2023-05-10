#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_train_logitnorm.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_logitnorm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 0
