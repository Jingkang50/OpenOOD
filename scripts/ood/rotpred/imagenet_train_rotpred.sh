#!/bin/bash
# sh scripts/ood/rotpred/imagenet_train_rotpred.sh

# batch size is 64 otherwise will run out of GPU memory
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/rot_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.backbone.name resnet50 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
    --trainer.name rotpred \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 30 \
    --dataset.train.batch_size 64 \
    --num_gpus 2 --num_workers 16 \
    --merge_option merge \
    --seed 0
