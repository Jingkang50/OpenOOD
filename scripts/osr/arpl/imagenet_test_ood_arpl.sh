#!/bin/bash
# sh scripts/osr/arpl/imagenet_test_ood_arpl.sh

# NOTE!!!!
# need to manually change the checkpoint path in configs/pipelines/test/test_arpl.yml
SCHEME="fsood" # "ood" or "fsood"
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_${SCHEME}.yml \
    configs/networks/arpl_net.yml \
    configs/pipelines/test/test_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --network.feat_extract_network.name resnet50 \
    --num_workers 8 \
    --evaluator.ood_scheme ${SCHEME} \
    --seed 0
