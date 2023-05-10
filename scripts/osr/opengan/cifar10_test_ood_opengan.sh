#!/bin/bash
# sh scripts/osr/opengan/cifar10_test_ood_opengan.sh

# need to manually change the checkpoint path in test_opengan.yml
# corresponding to different runs
SEED=0
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/opengan.yml \
    configs/pipelines/test/test_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --num_workers 8 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --seed ${SEED}
