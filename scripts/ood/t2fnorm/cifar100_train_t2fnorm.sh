#!/bin/bash

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/t2fnorm_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_t2fnorm.yml \
    --seed $RANDOM \
