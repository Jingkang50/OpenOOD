#!/bin/bash
# sh scripts/ood/reweightood/cifar100_test_reweightood.sh

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_cider_net_reweightood_e100_lr0.1_default \
   --postprocessor reweightood \
   --save-score --save-csv
