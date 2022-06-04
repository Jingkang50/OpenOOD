#!/bin/bash
# sh scripts/0_basics/covid_train.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/covid/covid.yml \
configs/networks/resnet18_224x224.yml \
configs/pipelines/train/baseline.yml \
--optimizer.num_epochs 200 \
--optimizer.lr 0.0001 \
--optimizer.weight_decay 0.0005 \
--num_workers 8
