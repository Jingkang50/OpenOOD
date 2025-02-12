#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att pgd \
    --batch-size 50