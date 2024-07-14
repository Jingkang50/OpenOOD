#!/bin/bash
# sh scripts/ood/ish/imagenet_train_ish.sh
# pretrained model: https://drive.google.com/file/d/1EQimcdbJsKdU2uw4-BrqZO6tu4kXKtbG

python main.py  \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/train/train_ish.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./checkpoints/resnet50-0676ba61.pth \
    --trainer.trainer_args.param 0.85 \
    --optimizer.lr 0.003 \
    --optimizer.weight_decay_fc 0.00005 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --num_gpus 4 --num_workers 4 \
    --merge_option merge \
    --seed ${SEED}
