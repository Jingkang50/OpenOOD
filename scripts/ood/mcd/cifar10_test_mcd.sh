#!/bin/bash
# sh scripts/ood/mcd/cifar10_test_mcd.sh

# need to manually change the checkpoint path from each run
# can't use best.ckpt because mcd only trains for the last 10 epochs
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/mcd.yml \
    --network.backbone.name resnet18_32x32 \
    --network.pretrained True \
    --network.checkpoint 'results/cifar10_oe_mcd_mcd_e100_lr0.1_default/s0/last_epoch100_acc0.9420.ckpt' \
    --num_workers 8 \
    --seed 0
