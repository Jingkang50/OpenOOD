#!/bin/bash
# sh scripts/ood/adascale_l/imagenet200_test_ood_scale.sh

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor adascale_l \
    --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor adascale_l \
    --save-score --save-csv --fsood
