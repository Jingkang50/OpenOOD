#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att mpgd \
    --batch-size 50

