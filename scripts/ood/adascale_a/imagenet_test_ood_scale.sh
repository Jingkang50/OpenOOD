#!/bin/bash
# sh scripts/ood/adascale_a/imagenet_test_ood_scale.sh

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor adascale_a \
    --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor adascale_a \
    --save-score --save-csv --fsood
