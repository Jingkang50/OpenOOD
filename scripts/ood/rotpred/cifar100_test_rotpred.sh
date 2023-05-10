#!/bin/bash
# sh scripts/ood/rotpred/cifar100_test_rotpred.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_rot_net_rotpred_e100_lr0.1_default \
   --postprocessor rotpred \
   --save-score --save-csv
