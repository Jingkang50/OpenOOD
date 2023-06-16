#!/bin/bash
# sh scripts/ood/rotpred/imagenet200_test_rotpred.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_rot_net_rotpred_e90_lr0.1_default \
   --postprocessor rotpred \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_rot_net_rotpred_e90_lr0.1_default \
   --postprocessor rotpred \
   --save-score --save-csv --fsood
