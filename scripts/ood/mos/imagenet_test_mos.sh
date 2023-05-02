#!/bin/bash
# sh scripts/ood/mos/imagenet_test_mos.sh

SEED=0

# ood
python main.py \
    --config configs/datasets/imagenet/imagenet_double_label.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_mos.yml \
    configs/postprocessors/mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint results/imagenet_double_label_resnet50_mos_e5_lr0.003/s0/best.ckpt \
    --num_workers 8 \
    --seed 0

# full-spectrum ood
python main.py \
    --config configs/datasets/imagenet/imagenet_double_label.yml \
    configs/datasets/imagenet/imagenet_double_label_fsood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_mos.yml \
    configs/postprocessors/mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint results/imagenet_double_label_resnet50_mos_e5_lr0.003/s0/best.ckpt \
    --num_workers 8 \
    --seed 0 \
    --evaluator.ood_scheme fsood
