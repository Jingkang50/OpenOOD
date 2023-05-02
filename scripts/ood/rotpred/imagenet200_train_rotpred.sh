#!/bin/bash
# sh scripts/ood/rotpred/imagenet200_train_rotpred.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/rot_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name resnet18_224x224 \
    --trainer.name rotpred \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
