#!/bin/bash
# sh scripts/ad/cutpaste/cifar100_train_cutpaste.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/train/train_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.pretrained False \
--network.backbone.name resnet18_32x32 \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/checkpoints/cifar100_res18_acc78.20.ckpt' \
--num_workers 8 \
--optimizer.num_epochs 100 \
--merge_option merge
