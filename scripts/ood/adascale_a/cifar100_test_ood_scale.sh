#!/bin/bash
# sh scripts/ood/adascale_a/cifar100_test_ood_scale.sh

python scripts/eval_ood.py \
    --id-data cifar100 \
    --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor adascale_a \
    --save-score --save-csv
