#!/bin/bash
# sh scripts/uncertainty/regmixup/cifar10_train_regmixup.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_regmixup.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.trainer_args.alpha 20 \
    --seed 0
