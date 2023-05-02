#!/bin/bash
# sh scripts/uncertainty/randaugment/imagenet_test_ood_msp.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
   --ckpt-path ./results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt \
   --arch resnet50 \
   --postprocessor msp \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
   --ckpt-path ./results/imagenet_resnet50_base_e30_lr0.001_randaugment-2-9/s0/best.ckpt \
   --arch resnet50 \
   --postprocessor msp \
   --save-score --save-csv --fsood
