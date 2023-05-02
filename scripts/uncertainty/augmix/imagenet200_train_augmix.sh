#!/bin/bash
# sh scripts/uncertainty/augmix/imagenet200_train_augmix.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_augmix.yml \
    configs/preprocessors/augmix_preprocessor.yml \
    --dataset.train.dataset_class ImglistAugMixDataset \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
