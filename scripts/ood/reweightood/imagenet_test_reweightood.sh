#!/bin/bash
# sh scripts/ood/reweightood/imagenet_test_reweightood.sh

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_reweightood_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor reweightood \
  --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_reweightood_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor reweightood \
  --save-score --save-csv --fsood
