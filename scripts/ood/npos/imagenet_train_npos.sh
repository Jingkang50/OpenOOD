#!/bin/bash
# sh scripts/ood/npos/imagenet_train_npos.sh

# NPOS trainer cannot work with multiple GPUs (DDP) currently
# we observed CUDA OOM error on Quadro RTX 6000 24GB GPU
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/npos_net.yml \
    configs/pipelines/train/train_npos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 128 \
    --trainer.trainer_args.temp 0.1 \
    --trainer.trainer_args.sample_from 1000 \
    --trainer.trainer_args.K 400 \
    --trainer.trainer_args.cov_mat 0.1 \
    --trainer.trainer_args.start_epoch_KNN 1 \
    --trainer.trainer_args.ID_points_num 300 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0
