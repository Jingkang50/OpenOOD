#!/bin/bash
# sh scripts/ood/mixoe/imagenet200_test_mixoe.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_oe_resnet18_224x224_mixoe_e10_lr0.001_alpha0.1_beta1.0_cutmix_lam1.0_default \
   --postprocessor msp \
   --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_oe_resnet18_224x224_mixoe_e10_lr0.001_alpha0.1_beta1.0_cutmix_lam1.0_default \
   --postprocessor msp \
   --save-score --save-csv --fsood
