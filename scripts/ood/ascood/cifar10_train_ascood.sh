#!/bin/bash

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/ascood_net.yml \
    configs/pipelines/train/train_ascood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.trainer_args.ood_type shuffle \
    --trainer.trainer_args.p_inv 0.15 "$@"
