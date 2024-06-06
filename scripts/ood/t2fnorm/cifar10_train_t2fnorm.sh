#!/bin/bash

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/t2fnorm_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_t2fnorm.yml \
    --seed $RANDOM \
