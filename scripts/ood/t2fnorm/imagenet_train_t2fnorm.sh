#!/bin/bash

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/t2fnorm_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/train_t2fnorm.yml \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
