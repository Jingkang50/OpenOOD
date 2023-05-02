#!/bin/bash
# sh scripts/ood/cider/imagenet_test_cider.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_cider_e10_lr0.001_protom0.95_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor cider \
  --save-score --save-csv #--fsood


# full-spectrum ood
python scripts/eval_ood_imagenet.py \
  --ckpt-path ./results/imagenet_cider_net_cider_e10_lr0.001_protom0.95_default/s0/best.ckpt \
  --arch resnet50 \
  --postprocessor cider \
  --save-score --save-csv --fsood
