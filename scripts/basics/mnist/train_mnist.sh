#!/bin/bash
# sh scripts/basics/mnist/train_mnist.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 100 \
--num_workers 8 \
--mark osr_model
