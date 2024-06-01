#!/bin/bash
# sh scripts/ood/palm/cifar10_train_palm.sh


python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/palm_net.yml \
    configs/pipelines/train/train_palm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name palm \
    --network.backbone.name resnet18_32x32 \
    --dataset.train.batch_size 16 \
    --trainer.trainer_args.proto_m 0.9\
    --optimizer.lr 0.1 \
    --num_workers 8 \
    --optimizer.num_epochs 100 \
    --seed 0
