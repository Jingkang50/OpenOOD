#!/bin/bash
# sh scripts/ood/oe/imagenet200_train_oe.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_oe.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_oe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
