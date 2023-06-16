#!/bin/bash
# sh scripts/osr/opengan/cifar100_test_ood_opengan.sh

# NOTE!!!!
# need to manually change the network checkpoint path (not backbone) in configs/pipelines/test/test_opengan.yml
# corresponding to different runs
SEED=0
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/opengan.yml \
    configs/pipelines/test/test_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --num_workers 8 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --seed ${SEED}
