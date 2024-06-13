#!/bin/bash
# sh scripts/ood/reweightood/imagenet200_train_reweightood.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_reweightood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --dataset.train.batch_size 512 \
    --num_gpus 2 --num_workers 16 \
    --optimizer.num_epochs 90 \
    --seed $RANDOM
