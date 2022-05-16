#!/bin/bash
# sh scripts/c_ood/test_mos.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/MOS/ILSVRC-2012.yml \
configs/networks/test_mos.yml \
configs/pipelines/test/test_mos.yml \
configs/postprocessors/mos.yml \
--num_workers 1
