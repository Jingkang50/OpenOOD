#!/bin/bash
# sh scripts/ood/iodin/imagenet200_test_ood_odin.sh

# ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor iodin \
    --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood.py \
    --id-data imagenet200 \
    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --postprocessor iodin \
    --save-score --save-csv --fsood
