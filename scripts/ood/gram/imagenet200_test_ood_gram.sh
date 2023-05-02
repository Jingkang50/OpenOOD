#!/bin/bash
# sh scripts/ood/ebo/imagenet200_test_ood_gram.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor gram \
    --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor gram \
    --save-score --save-csv --fsood
