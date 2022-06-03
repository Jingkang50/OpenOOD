#!/bin/bash
# sh scripts/a_anomaly/3_patchcore.sh

# GPU=1
# CPU=1
# node=30
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/draem/bottle.yml \
configs/networks/patchcorenet.yml \
configs/pipelines/test/test_patchcore.yml \
configs/postprocessors/patch.yml \
configs/preprocessors/base_preprocessor.yml \
--evaluator.name ad \
--num_workers 4 \
--force_merge True
