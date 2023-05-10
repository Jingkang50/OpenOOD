#!/bin/bash
# sh scripts/osr/opengan/cifar10_train_opengan.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_opengan_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/opengan.yml \
    configs/pipelines/train/train_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --dataset.feat_root ./results/cifar10_resnet18_32x32_feat_extract_opengan_default/s${SEED} \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --seed ${SEED}
