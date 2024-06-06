#!/bin/bash

python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_t2fnorm_net_t2fnorm_e90_lr0.1_alpha0.1_default \
   --postprocessor t2fnorm \
   --save-score --save-csv
