#!/bin/bash
# sh scripts/ood/rotpred/cifar10_train_rotpred.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/rot_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.name rotpred \
    --seed 0
