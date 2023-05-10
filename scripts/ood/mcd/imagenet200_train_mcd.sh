#!/bin/bash
# sh scripts/ood/mcd/imagenet200_train_mcd.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_oe.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mcd.yml \
    --network.backbone.name resnet18_224x224 \
    --network.pretrained False \
    --trainer.start_epoch_ft 80 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
