#!/bin/bash
# sh scripts/ood/npos/imagenet200_train_npos.sh

# NPOS trainer cannot work with multiple GPUs (DDP) currently
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/npos_net.yml \
    configs/pipelines/train/train_npos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --dataset.train.batch_size 256 \
    --trainer.trainer_args.temp 0.1 \
    --trainer.trainer_args.sample_from 1000 \
    --trainer.trainer_args.K 400 \
    --trainer.trainer_args.cov_mat 0.1 \
    --trainer.trainer_args.start_epoch_KNN 40 \
    --trainer.trainer_args.ID_points_num 300 \
    --optimizer.num_epochs 90 \
    --optimizer.lr 0.1 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0
