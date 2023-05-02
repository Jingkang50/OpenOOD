#!/bin/bash
# sh scripts/uncertainty/deepaugment/imagenet200_train_deepaugment.sh

# the model sees three times the data as the baseline
# so only trains for 90/3=30 epochs
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet200/train_imagenet200_deepaugment.txt \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0 \
    --mark deepaugment
