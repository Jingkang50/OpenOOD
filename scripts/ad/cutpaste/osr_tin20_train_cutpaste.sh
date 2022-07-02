#!/bin/bash
# sh scripts/ad/cutpaste/tin20_train_cutpaste.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_tin20/tin20_seed1.yml \
configs/datasets/osr_tin20/tin20_seed1_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/train/train_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.pretrained False \
--network.backbone.name resnet18_64x64 \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/checkpoints/osr/tin20_seed1_acc77.23.ckpt' \
--num_workers 8 \
--optimizer.num_epochs 100 \
--merge_option merge
