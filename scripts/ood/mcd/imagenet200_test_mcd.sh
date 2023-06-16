#!/bin/bash
# sh scripts/ood/mcd/imagenet200_test_mcd.sh

# NOTE!!!!
# need to manually change the checkpoint path
# remember to use the last_*.ckpt because mcd only trains for the last 10 epochs
# and the best.ckpt (according to accuracy) is typically not within the last 10 epochs
# therefore using best.ckpt is equivalent to early stopping with standard cross-entropy loss
SCHEME="ood" # "ood" or "fsood"
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_${SCHEME}.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/mcd.yml \
    --network.backbone.name resnet18_224x224 \
    --network.pretrained True \
    --network.checkpoint 'results/imagenet200_oe_mcd_mcd_e90_lr0.1_default/s0/last_epoch90_acc0.8410.ckpt' \
    --num_workers 8 \
    --evaluator.ood_scheme ${SCHEME} \
    --seed 0
