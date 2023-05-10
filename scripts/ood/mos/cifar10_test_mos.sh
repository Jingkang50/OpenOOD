#!/bin/bash
# sh scripts/ood/mos/cifar10_test_mos.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

SEED=0
python main.py \
    --config configs/datasets/cifar10/cifar10_double_label.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_mos.yml \
    configs/postprocessors/mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint results/cifar10_double_label_resnet18_32x32_mos_e30_lr0.003/s${SEED}/best.ckpt \
    --num_workers 8 \
    --seed ${SEED}
