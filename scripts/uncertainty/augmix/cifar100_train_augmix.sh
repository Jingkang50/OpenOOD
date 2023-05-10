#!/bin/bash
# sh scripts/uncertainty/augmix/cifar10_train_augmix.sh

# somehow the loss will diverge to NaN if using JSD
# so just use no-jsd here
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_augmix.yml \
    configs/preprocessors/augmix_preprocessor.yml \
    --preprocessor.severity 3 \
    --trainer.trainer_args.jsd False \
    --dataset.train.dataset_class ImglistDataset \
    --optimizer.num_epochs 100 \
    --dataset.train.batch_size 128 \
    --seed 0 \
    --mark no-jsd
