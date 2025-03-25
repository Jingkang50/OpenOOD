#!/bin/bash

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att df \
    --batch-size 50

