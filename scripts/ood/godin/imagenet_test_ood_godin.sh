#!/bin/bash
# sh scripts/ood/godin/imagenet_test_ood_godin.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
python scripts/eval_ood_imagenet.py \
   --ckpt-path ./results/imagenet_godin_net_godin_e30_lr0.001_default/s0/best.ckpt \
   --arch resnet50 \
   --postprocessor godin \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
   --ckpt-path ./results/imagenet_godin_net_godin_e30_lr0.001_default/s0/best.ckpt \
   --arch resnet50 \
   --postprocessor godin \
   --save-score --save-csv --fsood
