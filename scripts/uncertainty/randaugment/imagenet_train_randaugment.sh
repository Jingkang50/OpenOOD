#!/bin/bash
# sh scripts/uncertainty/randaugment/imagenet_train_randaugment.sh

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/randaugment_preprocessor.yml \
    --preprocessor.n 2 \
    --preprocessor.m 9 \
    --network.pretrained True \
    --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0 \
    --mark randaugment-2-9
