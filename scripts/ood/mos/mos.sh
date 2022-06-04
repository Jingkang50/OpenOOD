#!/bin/bash
# sh scripts/c_ood/mos.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/MOS/ILSVRC-2012.yml \
configs/networks/mos.yml \
configs/pipelines/train/train_mos.yml \
configs/postprocessors/mos.yml \
--num_workers 1
