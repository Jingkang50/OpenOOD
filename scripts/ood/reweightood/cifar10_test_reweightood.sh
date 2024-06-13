#!/bin/bash
# sh scripts/ood/reweightood/cifar10_test_reweightood.sh

python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_cider_net_reweightood_e100_lr0.1_default \
   --postprocessor reweightood \
   --save-score --save-csv
