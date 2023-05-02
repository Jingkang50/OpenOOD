#!/bin/bash
# sh scripts/ood/cider/imagenet200_train_cider.sh

SEED=0
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/cider_net.yml \
    configs/pipelines/train/train_cider.yml \
    configs/preprocessors/base_preprocessor.yml \
    --preprocessor.name cider \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.lr 0.01 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 512 \
    --trainer.trainer_args.proto_m 0.95 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed ${SEED}
