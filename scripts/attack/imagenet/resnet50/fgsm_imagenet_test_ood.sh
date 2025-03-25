#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att fgsm \
    --batch-size 50

