#!/bin/bash
# sh scripts/uncertainty/randaugment/cifar10_train_randaugment.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/randaugment_preprocessor.yml \
    --seed 0 \
    --mark randaugment-1-14
