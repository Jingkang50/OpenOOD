#!/bin/bash
# sh scripts/ood/rotpred/cifar100_train_rotpred.sh

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/rot_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.name rotpred \
    --seed 0
