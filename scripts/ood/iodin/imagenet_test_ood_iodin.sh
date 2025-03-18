#!/bin/bash
# sh scripts/ood/iodin/imagenet_test_ood_odin.sh

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor iodin \
   --save-score --save-csv

# full-spectrum ood
python scripts/eval_ood_imagenet.py \
   --tvs-pretrained \
   --arch resnet50 \
   --postprocessor iodin \
   --save-score --save-csv --fsood
