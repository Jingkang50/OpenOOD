#!/bin/bash
# sh scripts/basics/osr_mnist6/train_mnist6.sh

GPU=1
CPU=1
node=63
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/networks/lenet.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/baseline.yml \
--network.pretrained False \
--dataset.image_size 28 \
--optimizer.num_epochs 100 \
--num_workers 4 \
--mark ensemble_5
