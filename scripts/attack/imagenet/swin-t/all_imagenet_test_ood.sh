#!/bin/bash


python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att pgd \
    --batch-size 50

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att fgsm \
    --batch-size 50

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att df \
    --batch-size 50

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch swin-t \
    --att mpgd \
    --batch-size 50



