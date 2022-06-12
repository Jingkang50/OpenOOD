#!/bin/bash
# sh scripts/basics/cifar10/train_cifar10_dist.sh

GPU=1
CPU=1
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/baseline.yml \
--dataset.image_size 32 \
--optimizer.num_epochs 100 \
--num_workers 8 \
--num_gpus 2 \
--num_machines 1 \
--machine_rank 0 \
--mark 0 &
