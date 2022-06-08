#!/bin/bash
# sh scripts/osr/opengan/cifar100_train_opengan.sh

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
configs/networks/opengan.yml \
configs/pipelines/train/train_opengan.yml \
configs/preprocessors/base_preprocessor.yml \
--num_workers 8 \
--dataset.feat_path 'results/cifar100_resnet18_32x32_feat_extract/feature.npz' \
--network.backbone.pretrained True \
--network.backbone.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1/best.ckpt'
