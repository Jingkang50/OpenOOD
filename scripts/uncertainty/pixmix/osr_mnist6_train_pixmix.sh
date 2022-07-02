#!/bin/bash
# sh scripts/uncertainty/pixmix/osr_mnist6_train_pixmix.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

CUDA_VISIBLE_DEVICES=1 python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml \
configs/preprocessors/pixmix_preprocessor.yml \
--dataset.train.batch_size 4096 \
--num_workers 0 \
--optimizer.num_epochs 100 \
--mark pixmix \
--merge_option merge
