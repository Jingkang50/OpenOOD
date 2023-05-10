#!/bin/bash
# sh scripts/ood/mos/imagenet200_test_mos.sh

SEED=0

# ood
python main.py \
    --config configs/datasets/imagenet200/imagenet200_double_label.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_mos.yml \
    configs/postprocessors/mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint results/imagenet200_double_label_resnet18_224x224_mos_e10_lr0.003/s${SEED}/best.ckpt \
    --num_workers 8 \
    --seed ${SEED}

# full-spectrum ood
python main.py \
    --config configs/datasets/imagenet200/imagenet200_double_label.yml \
    configs/datasets/imagenet200/imagenet200_double_label_fsood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_mos.yml \
    configs/postprocessors/mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint results/imagenet200_double_label_resnet18_224x224_mos_e10_lr0.003/s${SEED}/best.ckpt \
    --evaluator.ood_scheme fsood \
    --num_workers 8 \
    --seed ${SEED}
