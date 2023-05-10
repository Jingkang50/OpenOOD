#!/bin/bash
# sh scripts/ood/mos/cifar100_train_mos.sh

# GPU=1
# CPU=0


# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \

SEED=0
python main.py \
    --config configs/datasets/cifar100/cifar100_double_label.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.num_epochs 30 \
    --merge_option merge \
    --seed ${SEED}
