#!/bin/bash
# sh scripts/ood/mos/imagenet200_train_mos.sh

SEED=0
python main.py \
    --config configs/datasets/imagenet200/imagenet200_double_label.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed ${SEED}
