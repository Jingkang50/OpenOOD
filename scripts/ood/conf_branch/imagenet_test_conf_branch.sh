#!/bin/bash
# sh scripts/ood/conf_branch/imagenet_test_conf_branch.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# ood
python scripts/eval_ood_imagenet.py \
    --ckpt-path ./results/imagenet_conf_branch_net_conf_branch_e30_lr0.001_default/s0/best.ckpt \
    --arch resnet50 \
    --postprocessor conf_branch \
    --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
    --ckpt-path ./results/imagenet_conf_branch_net_conf_branch_e30_lr0.001_default/s0/best.ckpt \
    --arch resnet50 \
    --postprocessor conf_branch \
    --save-score --save-csv --fsood
