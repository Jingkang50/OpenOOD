#!/bin/bash
# sh scripts/ood/cider/cifar100_test_cider.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar100 \
   --root ./results/cifar100_cider_net_cider_e100_lr0.5_protom0.5_default \
   --postprocessor cider \
   --save-score --save-csv
