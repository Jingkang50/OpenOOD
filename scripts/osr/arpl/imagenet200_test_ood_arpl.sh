#!/bin/bash
# sh scripts/osr/arpl/imagenet200_test_ood_arpl.sh

# NOTE!!!!
# need to manually change the checkpoint path in configs/pipelines/test/test_arpl.yml
SCHEME="ood" # "ood" or "fsood"
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_${SCHEME}.yml \
    configs/networks/arpl_net.yml \
    configs/pipelines/test/test_arpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --network.feat_extract_network.name resnet18_224x224 \
    --num_workers 8 \
    --evaluator.ood_scheme ${SCHEME} \
    --seed 0
