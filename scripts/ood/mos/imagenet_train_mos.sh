#!/bin/bash
# sh scripts/ood/mos/imagenet_train_mos.sh

python main.py \
    --config configs/datasets/imagenet/imagenet_double_label.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/train_mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.num_epochs 5 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
