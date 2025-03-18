#!/bin/bash
# sh scripts/attack/pgd_cifar10_test_ood.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/attack_ood_imagenet.py
# especially if you want to get results from
# multiple runs
python scripts/attack_ood_imagenet.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor scale \
    --save-score --save-csv
