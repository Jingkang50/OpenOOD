#!/bin/bash
# sh scripts/osr/opengan/imagenet200_train_opengan.sh

SEED=0

# feature extraction
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_opengan_feat_extract.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.checkpoint "./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt" \
    --seed ${SEED}

# train
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/opengan.yml \
    configs/pipelines/train/train_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --dataset.feat_root ./results/imagenet200_resnet18_224x224_feat_extract_opengan_default/s${SEED} \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.num_epochs 90 \
    --seed ${SEED}
