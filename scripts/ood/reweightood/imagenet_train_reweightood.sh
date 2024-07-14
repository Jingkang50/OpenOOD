#!/bin/bash
# sh scripts/ood/reweightood/imagenet_train_reweightood.sh

# Only layer4 of ResNet50 backbone is fine-tuned due to memory constraints.
# Set find_unused_parameter=True for multi-gpu training.
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_reweightood.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 512 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
