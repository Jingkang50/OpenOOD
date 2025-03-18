#!/bin/bash
# sh scripts/ood/iodin/cifar100_test_ood_odin.sh

python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
   --postprocessor iodin \
   --save-score --save-csv
