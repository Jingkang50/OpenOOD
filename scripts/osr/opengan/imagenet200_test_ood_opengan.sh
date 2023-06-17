#!/bin/bash
# sh scripts/osr/opengan/imagenet200_test_ood_opengan.sh

# NOTE!!!!
# need to manually change the network checkpoint path (not backbone) in configs/pipelines/test/test_opengan.yml
# corresponding to different runs
SEED=0
SCHEME="ood" # "ood" or "fsood"
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_${SCHEME}.yml \
    configs/networks/opengan.yml \
    configs/pipelines/test/test_opengan.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/opengan.yml \
    --num_workers 8 \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${SEED}/best.ckpt \
    --evaluator.ood_scheme ${SCHEME} \
    --seed ${SEED}
