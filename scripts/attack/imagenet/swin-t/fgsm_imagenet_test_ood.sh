#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att fgsm \
    --batch-size 50


