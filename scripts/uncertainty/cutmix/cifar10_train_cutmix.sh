#!/bin/bash
# sh scripts/uncertainty/cutmix/cifar10_train_cutmix.sh

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
--config configs/datasets/cifar10/cifar10.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/train_cutmix.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 8 \
--optimizer.num_epochs 100 \
--trainer.trainer_args.cutmix_prob 0.5 \
--mark cutmix
