#!/bin/bash
# sh scripts/osr/opengan/feature_extract.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/feat_extract.yml \
configs/preprocessors/base_preprocessor.yml \
--network.checkpoint "results/cifar100_resnet18_32x32_base_e100_lr0.1/best.ckpt" \
--pipeline.extract_target train \
--merge_option merge \
--num_workers 8 \
--mark 0
