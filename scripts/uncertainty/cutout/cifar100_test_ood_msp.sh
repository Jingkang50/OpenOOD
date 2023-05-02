#!/bin/bash
# sh scripts/uncertainty/cutout/cifar100_test_ood_msp.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_cutout-1-8 \
   --postprocessor msp \
   --save-score --save-csv
