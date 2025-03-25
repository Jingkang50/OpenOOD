#!/bin/bash


python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att pgd \
    --batch-size 50

# python scripts/attack_ood_imagenet.py \
#     --tvs-pretrained \
#     --arch resnet50 \
#     --att fgsm \
#     --batch-size 50

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att df \
    --batch-size 50

python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att mpgd \
    --batch-size 50



