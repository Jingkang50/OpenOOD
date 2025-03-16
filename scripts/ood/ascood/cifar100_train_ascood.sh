#!/bin/bash

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/ascood_net.yml \
    configs/pipelines/train/train_ascood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --optimizer.fc_lr_factor 0.05 \
    --trainer.trainer_args.w 5.0 \
    --trainer.trainer_args.ood_type shuffle "$@"
