#!/bin/bash
# sh scripts/ood/logitnorm/imagenet_test_logitnorm.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor msp \
  --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_resnet50_logitnorm_e30_lr0.001_alpha0.04_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor msp \
  --save-score --save-csv --fsood
