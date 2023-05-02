#!/bin/bash
# sh scripts/ood/udg/imagenet200_train_udg.sh

# UDG trainer cannot work with multiple GPUs currently
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_oe.yml \
    configs/networks/udg_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_udg.yml \
    configs/preprocessors/base_preprocessor.yml \
    --dataset.train.dataset_class UDGDataset \
    --dataset.oe.dataset_class UDGDataset \
    --network.backbone.name resnet18_224x224 \
    --network.pretrained False \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 256 \
    --dataset.oe.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0
