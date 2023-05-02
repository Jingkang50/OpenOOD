#!/bin/bash
# sh scripts/ood/vos/imagenet_train_vos.sh

# we observed CUDA OOM error on Quadro RTX 6000 24GB GPUs
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --network.pretrained True \
    --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --feature_dim 2048 \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed ${SEED}
