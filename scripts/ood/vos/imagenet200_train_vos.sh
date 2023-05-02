#!/bin/bash
# sh scripts/ood/vos/imagenet200_train_vos.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
