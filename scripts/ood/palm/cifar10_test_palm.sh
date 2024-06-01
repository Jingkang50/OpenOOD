#!/bin/bash
# sh scripts/ood/palm/cifar10_test_palm.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_palm_net_palm_e100_batch16_lr0.1_protom0.9_protos6_default \
   --postprocessor palm \
   --save-score --save-csv
