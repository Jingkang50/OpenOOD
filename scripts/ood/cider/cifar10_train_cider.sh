#!/bin/bash
# sh scripts/ood/cider/cifar10_train_cider.sh

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_cider.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_32x32 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.proto_m 0.95 \
    --num_workers 8 \
    --optimizer.num_epochs 100 \
    --seed 0
