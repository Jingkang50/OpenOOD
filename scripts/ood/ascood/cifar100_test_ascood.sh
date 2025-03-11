#!/bin/bash

python scripts/eval_ood.py \
   --id-data cifar100 \
   --wrapper-net ASCOODNet \
   --root ./results/cifar100_ascood_net_ascood_e100_lr0.1_w5.0_p0.1_otype_shuffle_alpha_10.0_10.0_kl_div_True_default \
   --save-score --save-csv --postprocessor scale #adascale_l, adascale_a
