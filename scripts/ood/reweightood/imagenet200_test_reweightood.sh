#!/bin/bash
# sh scripts/ood/reweightood/imagenet200_test_reweightood.sh

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_cider_net_reweightood_e90_lr0.1_default \
   --postprocessor reweightood \
   --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_cider_net_reweightood_e90_lr0.1_default \
   --postprocessor reweightood \
   --save-score --save-csv --fsood
