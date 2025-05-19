#!/bin/bash

# full-spectrum ood
python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att cw \
    --batch-size 50
    # --postprocessor scale \
    # --save-score --save-csv --fsood

