#!/bin/bash
# sh scripts/uncertainty/pixmix/imagenet200_train_pixmix.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/pixmix_preprocessor.yml \
    --preprocessor.preprocessor_args.aug_severity 1 \
    --preprocessor.preprocessor_args.beta 4 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 128 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed ${SEED} \
    --mark pixmix
