#!/bin/bash
# sh scripts/uncertainty/augmix/cifar100_test_ood_msp.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_augmix_e100_lr0.1_no-jsd \
   --postprocessor msp \
   --save-score --save-csv
