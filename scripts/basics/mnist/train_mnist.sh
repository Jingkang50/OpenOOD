#!/bin/bash
# sh scripts/basics/mnist/train_mnist.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml
