#!/bin/bash
# sh scripts/ood/rotpred/imagenet_test_rotpred.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_rot_net_rotpred_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor rotpred \
  --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_rot_net_rotpred_e30_lr0.001_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor rotpred \
  --save-score --save-csv --fsood
