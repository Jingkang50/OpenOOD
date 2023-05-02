#!/bin/bash
# sh scripts/uncertainty/regmixup/imagenet_train_regmixup.sh

python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/train_regmixup.yml \
    configs/preprocessors/base_preprocessor.yml \
    --trainer.trainer_args.alpha 10 \
    --network.pretrained True \
    --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
