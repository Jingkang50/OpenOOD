#!/bin/bash
# sh scripts/uncertainty/randaugment/imagenet200_train_randaugment.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/randaugment_preprocessor.yml \
    --preprocessor.m 10 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0 \
    --mark randaugment-1-10
