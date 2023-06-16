#!/bin/bash
# sh scripts/ood/npos/imagenet200_test_npos.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_npos_net_npos_e90_lr0.1_default \
   --postprocessor npos \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_npos_net_npos_e90_lr0.1_default \
   --postprocessor npos \
   --save-score --save-csv --fsood
