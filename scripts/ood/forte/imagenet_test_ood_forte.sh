#!/bin/bash
# sh scripts/ood/forte/imagenet_test_ood_forte.sh

# Forte is training-free and label-free: it uses frozen self-supervised
# backbones (CLIP, ViT-MSN, DINOv2) instead of a trained classifier, so
# there is no checkpoint to pass. It is a two-sample test evaluated with
# the unified evaluator only (see the docstring of the script below).
python scripts/eval_ood_forte.py \
    --id-data imagenet \
    --batch-size 256 \
    --seed 0 \
    --save-csv results/forte/imagenet_forte.csv
