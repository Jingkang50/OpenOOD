#!/bin/bash
# sh scripts/ood/conf_branch/imagenet200_test_conf_branch.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_conf_branch_net_conf_branch_e90_lr0.1_default \
    --postprocessor conf_branch \
    --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_conf_branch_net_conf_branch_e90_lr0.1_default \
    --postprocessor conf_branch \
    --save-score --save-csv --fsood
