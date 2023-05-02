#!/bin/bash
# sh scripts/osr/arpl/imagenet200_train_arpl.sh

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/arpl_net.yml \
    configs/pipelines/train/train_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.feat_extract_network.name resnet50 \
    --network.feat_extract_network.pretrained True \
    --network.feat_extract_network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
