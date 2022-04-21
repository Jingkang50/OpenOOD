#!/bin/bash
# sh scripts/c_ood/-1_cifar_train_godin.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/networks/godinnet.yml \
configs/pipelines/train/baseline.yml \
configs/postprocessors/godin.yml \
--dataset.image_size 32 \
--optimizer.num_epochs 5 \
--num_workers 8 \
--trainer.name 'godin' \
--force_merge True \
--dataset.train.batch_size 128 \
--optimizer.weight_decay 0.0005
