#!/bin/bash
# sh scripts/ood/adascale_a/cifar10_test_ood_scale.sh

python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor adascale_a \
    --save-score --save-csv
