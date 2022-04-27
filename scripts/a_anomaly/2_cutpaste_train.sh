#!/bin/bash
# sh scripts/a_anomaly/2_cutpaste_train.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mvtec/bottle.yml \
configs/networks/cutpaste.yml \
configs/pipelines/train/train_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
--dataset.image_size 28 \
--ood_dataset.image_size 28 \
--network.name cutpaste \
--trainer.name cutpaste \
--evaluator.name cutpaste \
--optimizer.num_epochs 100 \
--recorder.name cutpaste \
--num_workers 4
