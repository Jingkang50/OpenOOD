#!/bin/bash
# sh scripts/a_anomaly/3_patchcore.sh
GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/digits/MV.yml \
configs/networks/wide_resnet_50_2.yml \
configs/pipelines/test/test_patchcore.yml \
configs/postprocessors/patch.yml \
configs/preprocessors/None_preprocessor.yml \
--preprocessor.name base \
--num_workers 4
