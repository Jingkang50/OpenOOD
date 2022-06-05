#!/bin/bash
# sh scripts/ad/cutpaste/cifar10_test_ood_cutpaste.sh

# GPU=1
# CPU=1
# node=68
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.backbone.name resnet18_32x32 \
--num_workers 8 \
--network.checkpoint "results/cifar10_projectionNet_cutpaste_e1_lr0.03/best_epoch1_auroc0.5146.ckpt"
