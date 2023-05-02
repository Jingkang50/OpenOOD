#!/bin/bash
# sh scripts/uncertainty/augmix/imagenet200_test_ood_msp.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_augmix_e90_lr0.1_default \
    --postprocessor msp \
    --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_augmix_e90_lr0.1_default \
    --postprocessor msp \
    --save-score --save-csv --fsood
