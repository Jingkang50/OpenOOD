#!/bin/bash
# sh scripts/ood/mixoe/imagenet200_train_mixoe.sh

SEED=0
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_oe.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mixoe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed ${SEED}
