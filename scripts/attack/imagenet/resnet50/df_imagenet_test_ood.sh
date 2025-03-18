#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att df \
    --batch-size 50
