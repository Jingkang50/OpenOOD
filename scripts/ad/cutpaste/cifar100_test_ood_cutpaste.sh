#!/bin/bash
# sh scripts/ad/cutpaste/cifar100_test_ood_cutpaste.sh

GPU=1
CPU=1
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.backbone.name resnet18_32x32 \
--num_workers 8 \
--network.pretrained True \
--network.checkpoint "results/cifar100_projectionNet_cutpaste_e100_lr0.03/best.ckpt"
